for region in boolardy boston chajnantor chiang_mai effelsberg green_bank mauna_kea meerkat metsahovi minamimaki mount_graham narrabri ngari owens_valley pic_de_bure pico_veleta princeton qitai san_agustin san_basilio sierra_negra south_pole summit_camp teide thule; do
    python /users/tom/maria/data/scripts/compute_atmospheric_spectrum.py --region $region --tag v2
done

# for region in boolardy boston chajnantor chiang_mai effelsberg green_bank mauna_kea meerkat metsahovi minamimaki mount_graham narrabri ngari owens_valley pic_de_bure pico_veleta princeton qitai san_agustin san_basilio sierra_negra south_pole summit_camp teide thule; do
#     python /users/tom/maria/data/scripts/compute_atmospheric_spectrum.py --region $region --tag v2 --res low
# done