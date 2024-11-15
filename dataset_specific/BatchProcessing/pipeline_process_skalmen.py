from massimal.pipeline import PipelineProcessor
from pathlib import Path

datasets_dirs = [
    '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201520-nw_hsi',
    '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201552-nw_hsi',
   '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201640-nw_hsi',
    '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201709-nw_hsi',
    '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201736-nw_hsi',
    '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201815-se_hsi',
    '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201842-se_hsi',
]

datasets_dirs = [Path(p) for p in datasets_dirs]

# for dataset_dir in datasets_dirs:
#     pl = PipelineProcessor(dataset_dir)
#     # pl.georeference_glint_corrected_reflectance(pitch_offset=2,altitude_offset=18)
#     pl.mosaic_geotiffs()
#     del(pl)

for dataset_dir in datasets_dirs[-1:]:
    pl = PipelineProcessor(dataset_dir)
    pl.mosaic_geotiffs()
    del(pl)