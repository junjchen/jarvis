import open_images.open_image as o_im

# show this image
#one_image_id = '6010_1_2'
one_image_id = '6120_2_2'

# show image returns the image, variable can be '3', 'A', 'M' or 'P', depending on what set of bands you would like
# no second variable is equal to '3'
# if anyone wants I will create a variable that gives all bands at once
image = o_im.show_image_poly(one_image_id, '3')
#o_im.show_train_poly()



