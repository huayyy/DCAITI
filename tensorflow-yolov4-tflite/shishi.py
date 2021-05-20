from core.dataset import Dataset
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np

def main(_argv):
    flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
    flags.DEFINE_string('weights', './scripts/yolov4.weights', 'pretrained weights')
    flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')



    shishi = Dataset(FLAGS, is_training=True)

    line = shishi.annotations[1].split()
    #['/media/user/Source/Data/coco_dataset/coco/images/val2017/000000507042.jpg', '12,183,156,582,23', '43,348,369,599,23', '179,113,390,478,23']
    print("line:")
    print(line)
    image_path = line[0]
    #image = cv2.imread(image_path)

    bboxes = np.array(
        [list(map(float, box.split(","))) for box in line[1:]]
    )
    print(bboxes)
    print("------------")
    for box in line[1:]:
        print(box)



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass