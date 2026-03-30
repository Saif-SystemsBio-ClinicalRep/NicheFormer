import scanpy as sc
import squidpy as sq
import pandas as pd

adata = sc.datasets.visium_sge(sample_id='V1_Breast_Cancer_Block_A_Section_1')
adata.var_names_make_unique()

sc.pp.filter_cells(adata, min_counts=500)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs['pct_counts_mt'] < 15, :]

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sq.gr.spatial_neighbors(adata, n_rings=1, coord_type="grid", n_neighs=6)
sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=-1)

top_svgs = adata.uns['moranI'].head(10)
print(top_svgs)

adata.write("v1_breast_cancer_processed.h5ad")
