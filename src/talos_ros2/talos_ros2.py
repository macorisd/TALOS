import os
import rclpy
import rclpy.node
import sensor_msgs.msg

from segmentation_msgs.srv import SegmentImage
from segmentation_msgs.msg import SemanticInstance2D
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose

import torch
import numpy as np
from cv_bridge import CvBridge

from pipeline.pipeline_main import PipelineTALOS

class TalosRos2(rclpy.node.Node):
    def __init__(self):
        super().__init__('TALOS_ROS') # Detectron_ros
        
        self.publish_visualization = self.declare_parameter("publish_visualization", True).value 
        visualization_topic = self.declare_parameter("visualization_topic", "/talos/segmentedImage").value # "/detectron/segmentedImage"
        self.visualization_pub = self.create_publisher(sensor_msgs.msg.Image, visualization_topic, 1)

        self.cv_bridge = CvBridge()

        self.segment_image_srv =  self.create_service(SegmentImage, "/talos/segment", self.segment_image) # "/detectron/segment"
        self.talos_pipeline = PipelineTALOS()
        
        self._logger.info("Done setting up!")
        self._logger.info(f"Advertising service: {self.segment_image_srv.srv_name}")

    def segment_image(self, request, response):
        numpy_image = self.cv_bridge.imgmsg_to_cv2(request.image)

        img = torch.from_numpy(numpy_image)
        img = img.permute(2, 0, 1)  # HWC -> CHW
        if torch.cuda.is_available():
            img = img.cuda()
        inputs = [{"image": img}]
        self.predictor.eval()
        with torch.no_grad():
            outputs = self.predictor( inputs )[0]

        #outputs = self.predictor( numpy_image )
        results = outputs["instances"].to("cpu")

        if results.has("pred_masks"):
            masks = np.asarray(results.pred_masks)
        else:
            return response
        
        boxes = results.pred_boxes if results.has("pred_boxes") else []
        scores = results.scores
        
        #This field requires a modified version of detectron2. The official version does not output the scores of "losing" classes 
        scores_all_classes = self.normalize( results.all_scores[:, self.interest_classes] ) if results.has("all_scores") else None

        
        for i, bbox in enumerate(boxes):

            semantic_instance = SemanticInstance2D()

            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]] = 255

            semantic_instance.mask = self.cv_bridge.cv2_to_imgmsg(mask)

            if scores_all_classes is not None:
                semantic_instance.detection = self.set_multiclass_detection(scores_all_classes[i,:], bbox)
            else:
                semantic_instance.detection = self.set_singleclass_detection(self._class_names[results.pred_classes[i]], float(scores[i]), bbox)

            response.instances.append(semantic_instance)

        if self.publish_visualization:
            visualizer = Visualizer(numpy_image[:, :, ::-1], detectron2.data.MetadataCatalog.get(self.cfg.dataloader.train.dataset.names), scale=1.2)
            visualizer = visualizer.draw_instance_predictions(results)
            img = visualizer.get_image()[:, :, ::-1]

            image_msg_a = self.cv_bridge.cv2_to_imgmsg(img)
            self.visualization_pub.publish(image_msg_a)

        return response

    def normalize(self, scores):
        for i in range( list(scores.shape)[0] ):
            scores[i] /= sum(scores[i])
        return scores

    def set_singleclass_detection(self, class_name, score, bbox):

        detection = Detection2D()
        detection.bbox = BoundingBox2D()

        width = float(bbox[2]-bbox[0])
        height = float(bbox[3]-bbox[1])

        detection.bbox.center.position.x = float(bbox[0]) + width * 0.5 
        detection.bbox.center.position.y = float(bbox[1]) + height * 0.5
        detection.bbox.size_x = width
        detection.bbox.size_y = height
        detection.results = []
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = class_name
        hypothesis.hypothesis.score = score

        detection.results.append(hypothesis)

        return detection

def main(args=None):
    rclpy.init(args=args)
    node = TalosRos2()

    rclpy.spin(node)

    node.destroy_node()


if __name__ == '__main__':
    main()
