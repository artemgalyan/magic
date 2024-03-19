from datetime import datetime

import click
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq

from pandas import read_csv
from tqdm import tqdm

DTYPES = {
    'date_time': 'string',
    'site_name': 'uint8',
    'posa_continent': 'uint8',
    'user_location_country': 'uint8',
    'user_location_region': 'uint16',
    'user_location_city': 'uint16',
    'orig_destination_distance': 'float32',
    'user_id': 'uint32',
    'is_mobile': 'bool',
    'is_package': 'bool',
    'channel': 'uint8',
    'srch_ci': 'string',
    'srch_co': 'string',
    'srch_adults_cnt': 'uint8',
    'srch_children_cnt': 'uint8',
    'srch_rm_cnt': 'uint8',
    'srch_destination_id': 'uint16',
    'srch_destination_type_id': 'uint8',
    'is_booking': 'bool',
    'cnt': 'uint16',
    'hotel_continent': 'uint8',
    'hotel_country': 'uint8',
    'hotel_market': 'uint16',
    'hotel_cluster': 'uint8',
}

def get_pa_attribute(attr: str):
    if attr == 'bool':
        return pa.bool_()
    return getattr(pa, attr)()


def get_pa_scheme() -> pa.Schema:
    return pa.schema([(key, get_pa_attribute(value)) for key, value in DTYPES.items()])


def get_pandas_scheme() -> dict[str, str]:
    return DTYPES


def log(message: str) -> None:
    click.echo(f'[{datetime.now():%H:%M:%S}] {message}')


def using_dask(path: str, n: int, output_path: str) -> None:
    log('Start reading the file')
    df = dd.read_csv(path)
    log('Read the file, repartitioning it')
    df = df.repartition(npartitions=n)
    log('Saving...')
    df.to_parquet(output_path)
    log('Finished')


def using_parquet(path: str, n: int, output_path: str) -> None:
    with read_csv(path, chunksize=n, dtype=get_pandas_scheme()) as reader, \
         pq.ParquetWriter(output_path, schema=get_pa_scheme()) as writer:
        for chunk in tqdm(reader, desc='Converting'):
            table = pa.Table.from_pandas(chunk)
            writer.write_table(table)
    log('Done')


@click.command()
@click.argument('df_path', type=click.Path(exists=True), required=True)
@click.argument('output_path', type=click.Path(), required=True)
@click.option('-n', type=click.INT, default=-1)
@click.option('--converter', '-c', type=click.STRING, default='dask')
def main(df_path: str, output_path: str, n: int,
         converter: str) -> None:
    if not df_path.endswith('.csv'):
        print('1st argument df_path should be a csv file')
        return

    converters = {
        'dask': (using_dask, 16),
        'pa': (using_parquet, 2 ** 16 - 1)
    }

    if converter not in converters:
        print(f'Converter should be one of {list(converters.keys())}')
        return

    log(f'Using {converter} as converter')
    converter, n_default = converters[converter]
    if n <= 0:
        n = n_default

    converter(df_path, n, output_path)
