# jarvis

## Goal

Detect and describe the shape of following objects from the high resolution satellite image

1. Buildings - large building, residential, non-residential, fuel storage facility, fortified building
2. Misc. Manmade structures 
3. Road 
4. Track - poor/dirt/cart track, footpath/trail
5. Trees - woodland, hedgerows, groups of trees, standalone trees
6. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
7. Waterway 
8. Standing water
9. Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
10. Vehicle Small - small vehicle (car, van), motorbike

## Image file information

WorldView-3 products are delivered to the customer as relative **radiometrically corrected** image pixels. Their values are a
function of how much spectral radiance enters the telescope aperture and the instrument conversion of that radiation into a
digital signal. 

### Data sample counts

* Total images: 450 * 4
* Training data: 25
    * 6010_1_2, 6010_4_2, 6010_4_4
    * 6040_1_0, 6040_1_3, 6040_2_2, 6040_4_4
    * 6060_2_3
    * 6070_2_3
    * 6090_2_0
    * 6100_1_3, 6100_2_2, 6100_2_3
    * 6110_1_2, 6110_3_1, 6110_4_0
    * 6120_2_0, 6120_2_2
    * 6140_1_2, 6140_3_1
    * 6150_2_3
    * 6160_2_1
    * 6170_0_4, 6170_2_4, 6170_4_1

### Image files

| File | Size (Rows x Cols) | Bands | Resolution | Color depth |
| --- | --- | --- | --- | --- |
| xxx_x_x | 3349 x 3396| Red, Green, Blue | 0.31m | 11 bits |
| xxx_x_x_A | 134 x 136 | 8 SWIR Bands | 7.5m | 14 bits |
| xxx_x_x_M | 837 x 849 | 8 Multispectral Bands | 1.24m | 11 bits |
| xxx_x_x_P | 3348 x 3396 | Panchromatic, greyscale, single band | 0.31m | 11 bits |

## Sensor bands information

| Band | Type | Wavelength |
| --- | --- | --- |
| Panchromatic | Panchromatic | 450 - 800 nm |
| Coastal | Multispectral | 400 - 450 nm |
| Blue | Multispectral | 450 - 510 nm |
| Green | Multispectral | 510 - 580 nm |
| Yellow | Multispectral | 585 - 625 nm |
| Red | Multispectral | 630 - 690 nm |
| Red Edge | Multispectral | 705 - 745 nm |
| Near-IR1 | Multispectral | 770 - 895 nm |
| Near-IR2 | Multispectral | 860 - 1040 nm |
| SWIR-1 | SWIR | 1195 - 1225 nm |
| SWIR-2 | SWIR | 1550 - 1590 nm |
| SWIR-3 | SWIR | 1640 - 1680 nm |
| SWIR-4 | SWIR | 1710 - 1750 nm |
| SWIR-5 | SWIR | 2145 - 2185 nm |
| SWIR-6 | SWIR | 2185 - 2225 nm |
| SWIR-7 | SWIR | 2235 - 2285 nm |
| SWIR-8 | SWIR | 2295 - 2365 nm |

## Using Docker

Docker is a container engine that stabilizes the runtime environment. A Dockerfile is included in the project. And the image has been pushed to DockerHub. https://cloud.docker.com/swarm/junjchen90/repository/docker/junjchen90/jarvis-machine/general 

After installed Docker, run the following command in the cloned repo's directory:

```
docker run -ti --name app -v `pwd`:/app junjchen90/jarvis:latest
```

Is will start a bash and you're in the container. (The command pulls images from DockerHub, starts it as a container named "app" and mount the current working directory to container's /app directory)

launch jupyter
```
jupyter notebook --ip 0.0.0.0 --allow-root --NotebookApp.iopub_data_rate_limit=10000000000
```

## Classification methods

![classification methods](https://raw.githubusercontent.com/junjchen/jarvis/master/clsfy_methods.png)

## References

Abburu S, Golla S B. Satellite image classification methods and techniques: A review[J]. International journal of computer applications, 2015, 119(8).
