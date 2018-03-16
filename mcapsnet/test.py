"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 3/14/18
-- Time: 10:49 AM
"""

import numpy as np
if __name__ == '__main__':
    def get_coords_to_add(dataset_name):
        options = {'mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
                              [[8., 12.], [12., 12.], [16., 12.]],
                              [[8., 16.], [12., 16.], [16., 16.]]], 28.),
                   'fashion_mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
                                      [[8., 12.], [12., 12.], [16., 12.]],
                                      [[8., 16.], [12., 16.], [16., 16.]]], 28.),
                   'smallNORB': ([[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                                  [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                                  [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                                  [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]], 32.)
                   }
        coord_add, scale = options[dataset_name]

        coord_add = np.array(coord_add, dtype=np.float32) / scale
        return coord_add

    print(get_coords_to_add("mnist"))
    print(get_coords_to_add("fashion_mnist"))
    print(get_coords_to_add("smallNORB"))