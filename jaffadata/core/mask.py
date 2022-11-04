import operator
import re

import pandas.api.types as types


class FrameMask:
    def __init__(self, specs):
        self.specs = [FrameMask._parse(spec) for spec in specs.split(',')]

    def value(self, df):
        mask = True
        for key, value, op in self.specs:
            # Convert the value to the appropriate type
            if types.is_integer_dtype(df.dtypes[key]):
                value = int(value)
            elif types.is_numeric_dtype(df.dtypes[key]):
                value = float(value)

            mask &= op(df[key], value)
        return mask

    def _parse(spec):
        ops = {
            '<': operator.lt,
            '<=': operator.le,
            '=': operator.eq,
            '==': operator.eq,
            '!=': operator.ne,
            '>=': operator.ge,
            '>': operator.gt,
        }

        pattern = r'(\w+)\s*([!=<>]+)\s*(["\']?)([^"\'!=<>]+|-)?\3$'
        match = re.match(pattern, spec.strip())
        if match is None or ops.get(match[2]) is None:
            raise ValueError(f'Invalid mask specification: {spec}')

        # Return (key, value, operator)
        return match[1], match[4] or '', ops[match[2]]
