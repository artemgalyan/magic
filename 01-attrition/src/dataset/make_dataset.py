from pathlib import Path

import click

from pandas import get_dummies, read_csv


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def main(input_file: str, output_file: str) -> None:
    assert Path(input_file).exists(), f'{input_file} do not exist'
    df = read_csv(input_file) \
        .drop(columns=['Over18', 'EmployeeCount', 'StandardHours']) \
        .set_index('EmployeeID', drop=True, verify_integrity=True)
    df = get_dummies(df, drop_first=True)
    df.to_csv(output_file)
