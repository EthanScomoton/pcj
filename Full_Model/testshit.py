from All_Models_EGrid_Paper import load_data, feature_engineering
data_df   = load_data()
data_df, feature_cols, _ = feature_engineering(data_df)

orig_dim = 22
print("原始模型 22 个特征：")
for i, name in enumerate(feature_cols[:orig_dim], 1):
    print(f"{i:2d}. {name}")

if len(feature_cols) > orig_dim:
    print("\n新增特征 (共 %d 个):" % (len(feature_cols)-orig_dim))
    for i, name in enumerate(feature_cols[orig_dim:], orig_dim+1):
        print(f"{i:2d}. ★ {name}")