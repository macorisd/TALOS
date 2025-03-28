This directory should contain the output of the **object location stage**.

**Format**: 
- _.json_ file with a list of semantic tags.
- _.jpg_ image file with information of the object bounding box locations.

**Example of the _.json_ file**:

```json
[
    {
        "label": "table",
        "score": 0.8407365083694458,
        "bbox": {
            "x_min": 196.67405700683594,
            "y_min": 282.10223388671875,
            "x_max": 847.9370727539062,
            "y_max": 774.1167602539062
        }
    },
    {
        "label": "chair",
        "score": 0.2740229368209839,
        "bbox": {
            "x_min": 369.80731201171875,
            "y_min": 488.4823303222656,
            "x_max": 482.1817932128906,
            "y_max": 596.7686157226562
        }
    }
]
```