from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import click
from numpy import nan
from pandas import read_csv, DataFrame, Series, to_datetime, to_numeric

SECONDS_IN_HOUR = 3600


def preprocess_categorical_features(data: DataFrame) -> DataFrame:
    data['Education'] = data['Education'].replace(list(range(1, 6)),
                                                  ['Below College', 'College', 'Bachelor', 'Master', 'Doctor'])
    data['WorkLifeBalance'] = data['WorkLifeBalance'].replace(list(range(1, 5)), ['Bad', 'Good', 'Better', 'Best'])

    for column in ['EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating']:
        data[column] = data[column].replace(list(range(1, 5)), ['Low', 'Medium', 'High', 'Very High'])
    return data


def preprocess_time_dataframe(times: DataFrame) -> DataFrame:
    """
    Preprocesses the dataframe: replaces NA with None and converts all dates to time in seconds format
    :param times: input dataframe with in/out times
    :return: preprocessed dataframe
    """

    times[times == 'NA'] = nan
    for column in times.columns[1:]:
        times[column] = to_datetime(times[column]).map(
            lambda x: x if x is None else (x - x.replace(hour=0, minute=0, second=0)).total_seconds() / SECONDS_IN_HOUR)

    return times


def get_working_time_statistics(in_times_: DataFrame, out_times_: DataFrame, id_column: str = 'EmployeeID',
                                statistics: list[str] | None = None) -> Generator[Series, None, None]:
    """
    Calculates named statistics for working time
    :param in_times_: dataframe with working day start times
    :param out_times_: dataframe with working day end times
    :param statistics: statistics to calculate, should be methods of pandas.DataFrame and should result in Series object
    :return: Generator yielding series with given statistics
    """

    assert (in_times_.index == out_times_.index).all(), 'in_times and out_times index should match'
    assert (in_times_.columns == out_times_.columns).all(), 'in_times columns and out_times should be equal'

    if statistics is None:
        statistics = ['mean', 'median', 'skew']

    relevant_columns = in_times_.columns[1:]
    working_time = out_times_[relevant_columns] - in_times_[relevant_columns]

    for statistic_name in statistics:
        assert hasattr(working_time, statistic_name), \
            f'Statistic {statistic_name} can not be calculated for the dataframe, ensure dataframe has this method'
        statistic = getattr(working_time, statistic_name)(axis=1)
        assert isinstance(statistic, Series), f'dataframe.{statistic_name}() should be a series object'
        statistic.name = statistic_name.capitalize() + 'WorkingTime'
        statistic[id_column] = in_times_.index
        yield to_numeric(statistic, errors='coerce')


def log(message: str) -> None:
    click.echo(f'[{datetime.now():%H:%M:%S}] {message}')


def validate_file(raw_data_dir: Path, filename: str) -> None:
    assert filename.endswith('.csv'), f'{filename} file must be .csv file'
    assert (raw_data_dir / filename).exists(), f'{raw_data_dir / filename} must exist'


@click.command()
@click.argument('raw_data_dir', type=click.Path(exists=True), required=True)
@click.argument('output_path', type=click.Path(), required=True)
@click.argument('statistics', type=click.STRING, nargs=-1, required=True)
@click.option('--general-data', '-gd', type=click.STRING, default='general_data.csv')
@click.option('--empl-surv-data', '-esd', type=click.STRING, default='employee_survey_data.csv')
@click.option('--mngr-surv-data', '-msd', type=click.STRING, default='manager_survey_data.csv')
@click.option('--in-times', '-it', type=click.STRING, default='in_time.csv')
@click.option('--out-times', '-ot', type=click.STRING, default='out_time.csv')
@click.option('--id-column', '-id', type=click.STRING, default='EmployeeID')
def main(raw_data_dir: str, output_path: str, statistics: list[str],
                general_data: str, empl_surv_data: str,
                mngr_surv_data: str, in_times: str, out_times: str, id_column: str) -> None:
    raw_data_dir = Path(raw_data_dir)
    output_path = Path(output_path)
    for file in [general_data, empl_surv_data, mngr_surv_data, in_times, out_times]:
        validate_file(raw_data_dir, file)

    log('Reading raw data')
    general_data = read_csv(raw_data_dir / general_data)
    manager_survey_data = read_csv(raw_data_dir / mngr_surv_data)
    employee_survey_data = read_csv(raw_data_dir / empl_surv_data)
    for df in [general_data, manager_survey_data, employee_survey_data]:
        assert id_column in df.columns, f'Id column {id_column} should be in all dataframes except time ones'
    in_times = read_csv(raw_data_dir / in_times)
    out_times = read_csv(raw_data_dir / out_times)
    log('Data is loaded. Preprocessing the data')
    in_times = preprocess_time_dataframe(in_times)
    out_times = preprocess_time_dataframe(out_times)
    general_data.set_index(id_column, inplace=True)
    manager_survey_data.set_index(id_column, inplace=True)
    employee_survey_data.set_index(id_column, inplace=True)
    data = general_data \
        .join(manager_survey_data, on=id_column, how='left', lsuffix='', rsuffix='_r') \
        .join(employee_survey_data, on=id_column, how='left', lsuffix='', rsuffix='_r')
    data = preprocess_categorical_features(data)
    log('Adding features')
    for statistic in get_working_time_statistics(in_times, out_times, id_column, statistics):
        data = data.join(statistic, on=id_column, how='left')
    log('Saving the data')
    data.to_csv(output_path)
