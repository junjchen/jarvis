import logistic_regression_classifier as lrc
import read_image
import shapely

#IM_ID = '6120_2_2'
IM_ID = '6120_2_2'
POLY_TYPE = '5'  # buildings
[im_rgb, train_mask, im_size, x_scaler, y_scaler, train_polygons] = read_image.read_image(IM_ID, POLY_TYPE)
[xs, ys, pipeline] = lrc.setup(im_rgb, train_mask)
pred_binary_mask = lrc.train(xs, ys, pipeline, train_mask)
# check jaccard on the pixel level
tp, fp, fn = (( pred_binary_mask &  train_mask).sum(),
              ( pred_binary_mask & ~train_mask).sum(),
              (~pred_binary_mask &  train_mask).sum())
print('Pixel jaccard', tp / (tp + fp + fn))

pred_polygons = lrc.mask_to_polygons(pred_binary_mask)
pred_poly_mask = read_image.mask_for_polygons(pred_polygons, im_size)
read_image.show_mask(pred_poly_mask[2900:3200,2000:2300])

scaled_pred_polygons = shapely.affinity.scale(
    pred_polygons, xfact=1 / x_scaler, yfact=1 / y_scaler, origin=(0, 0, 0))

dumped_prediction = shapely.wkt.dumps(scaled_pred_polygons)
print('Prediction size: {:,} bytes'.format(len(dumped_prediction)))
final_polygons = shapely.wkt.loads(dumped_prediction)

print('Final jaccard',
      final_polygons.intersection(train_polygons).area /
      final_polygons.union(train_polygons).area)
