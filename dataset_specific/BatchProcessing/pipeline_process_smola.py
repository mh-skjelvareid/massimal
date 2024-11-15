from massimal.pipeline import PipelineProcessor
from pathlib import Path

datasets_dirs = [
    # '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201520-nw_hsi',
    # '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201552-nw_hsi',
#    '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201640-nw_hsi',
    # '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201709-nw_hsi',
    # '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201736-nw_hsi',
    # '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201815-se_hsi',
    # '/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201842-se_hsi',
    # '/media/mha114/Massimal2/seabee-minio/smola/maholmen/aerial/hsi/20230621/massimal_smola_maholmen_202306211129-2_hsi',
    '/media/mha114/Massimal2/seabee-minio/smola/maholmen/aerial/hsi/20230621/massimal_smola_maholmen_202306211155-2_hsi',
    '/media/mha114/Massimal2/seabee-minio/smola/maholmen/aerial/hsi/20230621/massimal_smola_maholmen_202306211228-2_hsi',
    '/media/mha114/Massimal2/seabee-minio/smola/maholmen/aerial/hsi/20230621/massimal_smola_maholmen_202306211324-2_hsi',
    '/media/mha114/Massimal2/seabee-minio/smola/maholmen/aerial/hsi/20230621/massimal_smola_maholmen_202306211355-3_hsi',
    '/media/mha114/Massimal2/seabee-minio/smola/maholmen/aerial/hsi/20230621/massimal_smola_maholmen_202306211432-3_hsi',
]

datasets_dirs = [Path(p) for p in datasets_dirs]

for dataset_dir in datasets_dirs:
    pl = PipelineProcessor(dataset_dir)
    pl.run()
    del(pl)