import pandas as pd

df = pd.read_pickle('matrix2.pkl')
opt = df['opt'].tolist()
opt_ret = []
compiler = df['compiler'].tolist()
compiler_ret = []
df = df.drop('opt', axis=1)
df = df.drop('compiler', axis=1)
for elem in opt:
    if 'L' in elem:
        opt_ret.append(0)
    else:
        opt_ret.append(1)

for elem in compiler:
    if 'gcc' in elem:
        compiler_ret.append(0)
    elif 'icc' in elem:
        compiler_ret.append(1)
    else:
        compiler_ret.append(2)

df_opt = pd.DataFrame(columns=['opt'], data=opt_ret)
df_compiler = pd.DataFrame(columns=['compiler'], data=compiler_ret)
df = pd.concat([df, df_opt, df_compiler], axis=1)
df.to_pickle('matrix_zero_ones.pkl')

