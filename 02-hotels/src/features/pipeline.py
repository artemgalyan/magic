from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl


DataFrame = pl.DataFrame


def map_to_polars(dtype: str):
    conversion = {
        'string': pl.String,
        'uint8': pl.UInt8,
        'uint16': pl.UInt16,
        'uint32': pl.UInt32,
        'int16': pl.Int16,
        'int32': pl.Int32,
        'float32': pl.Float32,
        'bool': pl.UInt8 
    }
    return conversion[dtype]


def map_to_np(dtype):
    conversion = {
        pl.String: 'string',
        pl.UInt8: 'uint8',
        pl.UInt16: 'uint16',
        pl.UInt32: 'uint32',
        pl.Float32: 'float32',
        pl.Int8: 'int8',
        pl.Int16: 'int16',
        pl.Int32: 'int32',
        pl.Boolean: 'bool'
    }
    return conversion.get(dtype, 'datetime')


@dataclass
class PipelineState:
    schema: dict[str, str] = field(default_factory=dict)
    numerical_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numerical_categorical_columns: list[str] = field(default_factory=list)


class PipelineProcessor(ABC):
    '''
    Single pipeline processor
    '''

    @abstractmethod
    def fit_transform(self, x: DataFrame, state: PipelineState) -> tuple[DataFrame, PipelineState]:
        pass

    @abstractmethod
    def transform(self, x: DataFrame) -> DataFrame:
        pass


class FeatureExtractorPipeline:
    def __init__(self, pipeline: list[PipelineProcessor]) -> None:
        self.pipeline = pipeline

    def fit_transform(self, df: DataFrame) -> tuple[DataFrame, PipelineState]:
        state = PipelineState(
            schema={
                k: map_to_np(v) for k, v in df.schema.items()
            }
        )
        for processor in self.pipeline:
            df, state = processor.fit_transform(df, state)
        return df, state


class AddColumns(PipelineProcessor):
    def __init__(self, expressions: dict[pl.expr, str]) -> None:
        self.expressions = expressions
    
    def fit_transform(self, x: pl.DataFrame, state: PipelineState) -> tuple[DataFrame, PipelineState]:
        for col, (_, dtype) in self.expressions.items():
            state.schema[col] = dtype
        return self.transform(x), state
    
    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        return x.with_columns(**{
            col: expr.cast(map_to_polars(dtype)) for col, (expr, dtype) in self.expressions.items() 
        })


class DropColumns(PipelineProcessor):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit_transform(self, x: pl.DataFrame, state: PipelineState) -> tuple[DataFrame, PipelineState]:
        state.schema = {k: v for k, v in state.schema.items() if k not in self.columns}
        return self.transform(x), state
    
    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        return x.drop(*self.columns)


class Cast(PipelineProcessor):
    def __init__(self, to: dict[str, str]) -> None:
        self.to = to
    
    def fit_transform(self, x: pl.DataFrame, state: PipelineState) -> tuple[DataFrame, PipelineState]:
        for col, dtype in self.to.items():
            state.schema[col] = dtype
        return self.transform(x), state
    
    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        return x.with_columns(**{
            col: pl.col(col).cast(map_to_polars(dtype)) for col, dtype in self.to.items()   
        })


class ColumnSplitter(PipelineProcessor):
    def __init__(self, num_cat_threshold: int = 0) -> None:
        self.threshold = num_cat_threshold
        self.numerical_categorical = []
        self.categorical = []
        self.numerical = []
    
    def fit_transform(self, x: pl.DataFrame, state: PipelineState) -> tuple[DataFrame, PipelineState]:
        state.numerical_categorical_columns = [
            col for col in x.columns
            if x[col].dtype not in [pl.Boolean, pl.String, pl.Datetime] and x[col].n_unique() < self.threshold
        ]
        state.numerical_columns = [
            col for col in x.columns
            if x[col].dtype not in [pl.Boolean, pl.String, pl.Datetime] and col not in state.numerical_categorical_columns
        ]
        state.categorical_columns = [
            col for col in x.columns
            if col not in state.numerical_columns and col not in state.numerical_categorical_columns
        ]
        self.numerical_categorical = state.numerical_categorical_columns.copy()
        self.categorical = state.categorical_columns.copy()
        self.numerical = state.numerical_columns.copy()
        return self.transform(x), state
    
    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        return x
