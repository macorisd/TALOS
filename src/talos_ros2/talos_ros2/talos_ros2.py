import numpy as np
import cv2

import rclpy
import rclpy.node
import sensor_msgs.msg
from cv_bridge import CvBridge

from segmentation_msgs.srv import SegmentImage
from segmentation_msgs.msg import SemanticInstance2D
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose

from talos_ros2.pipeline_ros2_main import PipelineTALOSRos2

class TALOSRos2Node(rclpy.node.Node):
    def __init__(self):
        super().__init__('talos_ros2_node')
        
        self.publish_visualization = self.declare_parameter("publish_visualization", True).value 
        visualization_topic = self.declare_parameter("visualization_topic", "/talos/segmentedImage").value # "/detectron/segmentedImage"
        self.visualization_pub = self.create_publisher(sensor_msgs.msg.Image, visualization_topic, 1)

        self.cv_bridge = CvBridge()

        self.segment_image_srv =  self.create_service(SegmentImage, "/talos/segment", self.segment_image) # "/detectron/segment"
        self.talos_pipeline = PipelineTALOSRos2()
        
        self._logger.info("Done setting up!")
        self._logger.info(f"Advertising service: {self.segment_image_srv.srv_name}")

    def segment_image(self, request, response):
        numpy_image = self.cv_bridge.imgmsg_to_cv2(request.image)

        segmentation_info, all_masks, _ = self.talos_pipeline.run(input_image=numpy_image)
        detections = segmentation_info.get("detections", [])

        for i, det in enumerate(detections):
            semantic_instance = SemanticInstance2D()

            # Get the mask and convert it to ROS Image message
            mask = all_masks[i].astype("uint8") * 255
            semantic_instance.mask = self.cv_bridge.cv2_to_imgmsg(mask)

            # Get bbox and label
            bbox = det.get("bbox", [0, 0, 0, 0])
            label = det.get("label", "unknown")

            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min

            detection = Detection2D()
            detection.bbox = BoundingBox2D()
            detection.bbox.center.position.x = x_min + width / 2.0
            detection.bbox.center.position.y = y_min + height / 2.0
            detection.bbox.size_x = width
            detection.bbox.size_y = height

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = label
            hypothesis.hypothesis.score = 1.0  # As TALOS doesn't provide scores, default to 1.0
            detection.results.append(hypothesis)

            semantic_instance.detection = detection
            response.instances.append(semantic_instance)

        # Visualization
        if self.publish_visualization:
            img_vis = numpy_image.copy()
            for i, det in enumerate(detections):
                bbox = det.get("bbox", [0, 0, 0, 0])
                label = det.get("label", "unknown")
                x_min, y_min, x_max, y_max = map(int, bbox)
                cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(img_vis, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Optionally overlay the mask
                img_vis[all_masks[i] > 0] = (img_vis[all_masks[i] > 0] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

            image_msg = self.cv_bridge.cv2_to_imgmsg(img_vis)
            self.visualization_pub.publish(image_msg)

        return response

    def normalize(self, scores):
        for i in range( list(scores.shape)[0] ):
            scores[i] /= sum(scores[i])
        return scores


def main(args=None):
    rclpy.init(args=args)
    node = TALOSRos2Node()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
