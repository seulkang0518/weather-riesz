"""Optional parity audit: execute preprocessing statements from the pinned script.

Requires scikit-learn in addition to requirements.txt. Does not import TensorFlow
or execute upstream training. Compares all 12 model arrays against our NPZ.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.compute as pc
import pyarrow as pa
from sklearn import preprocessing

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--prepared', default='data/temperature10_group0.npz')
    p.add_argument('--output', default='validation/preprocessing_parity.json')
    a = p.parse_args()
    meta = json.loads(Path(a.prepared).with_suffix('.json').read_text())
    dim, k = len(meta['stations']), meta['station_group']
    groups = pd.read_csv(ROOT/f'third_party/mvpp/t2m_dist_{dim}samples.csv', header=None)
    table = feather.read_table(a.source, memory_map=True)
    selected = pa.array(groups.iloc[k].astype(int).tolist(), type=table['station'].type)
    df = table.filter(pc.is_in(table['station'], value_set=selected)).to_pandas()
    del table
    source = (ROOT/'third_party/mvpp/CGM_mvpp_t2m.py').read_text()
    tree = ast.parse(source)
    loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id=='k')
    statements = []
    for node in loop.body:
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BATCH_SIZE'
                                               for t in node.targets)):
            break
        statements.append(node)
    code = ast.Module(body=statements, type_ignores=[])
    class PandasCompatibility(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            # Old pandas implicitly flattened (N,1) scaler results assigned to a
            # single iloc column. Pandas 3 rejects that. Shape-only adaptation;
            # the scaler operations and all values remain the upstream ones.
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == 'transform'):
                node.value = ast.Call(func=ast.Attribute(value=node.value, attr='reshape', ctx=ast.Load()),
                                      args=[ast.Constant(-1)], keywords=[])
            return node
    code = ast.fix_missing_locations(PandasCompatibility().visit(code))
    namespace = {'np': np, 'pd': pd, 'preprocessing': preprocessing,
                 'data_complete': df, 'dist_samples': groups, 'dim': dim, 'k': k}
    exec(compile(code, str(ROOT/'third_party/mvpp/CGM_mvpp_t2m.py'), 'exec'), namespace)
    prepared = np.load(a.prepared)
    errors = {}
    for split, upstream in [('train','training'), ('val','validation'), ('test','test')]:
        for i, key in enumerate(('mean', 'std', 'context')):
            actual = prepared[f'{split}_{key}']
            expected = namespace[f'x_{upstream}'][i].astype(np.float32)
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
            errors[f'{split}_{key}'] = float(np.max(np.abs(actual-expected)))
        np.testing.assert_array_equal(prepared[f'{split}_target'], namespace[f'y_{upstream}'][...,0].astype(np.float32))
    result = {'passed': True, 'official_preprocessing_executed_directly': True,
              'compatibility_change': 'flatten (N,1) scaler results before pandas column assignment',
              'max_abs_error': errors, 'targets_exact_float32': True,
              'scope': f'station group {k}; all train/validation/test rows'}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
