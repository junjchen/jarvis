import logistic_regression_classifier as lrc
import read_image
import shapely

#The ID of the image used for initial learning and testing
#IM_ID = '6120_2_2'
IM_ID = '6070_2_3'

#The type of polygons we learn and test on
#1: 'Bldg', 2: 'Struct',3: 'Road',4: 'Track',5: 'Trees',6: 'Crops',7: 'Fast H20',8: 'Slow H20',9: 'Truck',10: 'Car'.
POLY_TYPE = '7'

# Use read_image to get the rgb values, training mask, size, x and y scalers and the training polygons
#[im_rgb, train_mask, im_size, x_scaler, y_scaler, train_polygons] = read_image.read_image(IM_ID, POLY_TYPE)
#w, h = im_size
#print("Image size: h =" + str(h) + " w = " + str(w))

#lrc.stick_all_train()

#Use the lrc to create xs: inpits, ys: outputs and a training pipeline from the image and the training mask
#[xs, ys, pipeline] = lrc.setup(im_rgb, train_mask)
#setup with all train images
[xs, ys, pipeline] = lrc.setup()

#lrc.make_submit()
print('hello world')

#Use the inputs, outputs, pipeline and training mask to train a model and predict outputs
#pred_binary_mask = lrc.train(xs, ys, pipeline, train_mask)

# check jaccard on the pixel level from the first set of outputs
#tp, fp, fn = (( pred_binary_mask &  train_mask).sum(),
#              ( pred_binary_mask & ~train_mask).sum(),
#              (~pred_binary_mask &  train_mask).sum())
#print('Pixel jaccard', tp / (tp + fp + fn))

#convert the predicted pixel mask to polygons
#pred_polygons = lrc.mask_to_polygons(pred_binary_mask)

#make a polygon mask based on these polygons
#pred_poly_mask = read_image.mask_for_polygons(pred_polygons, im_size)

#show the trained mask
#read_image.show_mask(pred_poly_mask[2900:3200,2000:2300])
#read_image.show_mask(pred_poly_mask)

#Scale the predicted polygon mask from 1 by 1 to the image size
#scaled_pred_polygons = shapely.affinity.scale(
#    pred_polygons, xfact=1 / x_scaler, yfact=1 / y_scaler, origin=(0, 0, 0))

#make an output prediction from the scaled mask
#dumped_prediction = shapely.wkt.dumps(scaled_pred_polygons)
#print size of prediction
#print('Prediction size: {:,} bytes'.format(len(dumped_prediction)))
#load the output prediction
#final_polygons = shapely.wkt.loads(dumped_prediction)

#Find the output score
#print('Final jaccard',
#      final_polygons.intersection(train_polygons).area /
#      final_polygons.union(train_polygons).area)
