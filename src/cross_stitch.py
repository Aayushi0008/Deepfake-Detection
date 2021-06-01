import tensorflow as tf

class CrossStitch(tf.keras.layers.Layer):
    def __init__(self, num_tasks, *args, **kwargs):
        self.num_tasks = num_tasks
        super(CrossStitch, self).__init__(**kwargs)

    def build(self, input_shape):
        print("input shape", input_shape)
        self.kernel = self.add_weight(name="kernel",
                                      shape=(self.num_tasks,
                                             self.num_tasks),
                                      initializer=custom_init,
                                      trainable=True)
        super(CrossStitch, self).build(input_shape)

    def call(self, xl):
        output = []
        for this_task in range(self.num_tasks):
            this_weight = self.kernel[this_task, this_task]
            out = tf.math.scalar_mul(this_weight, xl[this_task])
            for other_task in range(self.num_tasks):
                if this_task == other_task:
                    continue
                    other_weight = self.kernel[this_task, other_task]
                    out += tf.math.scalar_mul(other_weight, xl[other_task])
            output.append(out)

        return tf.stack(output, axis=0)

    def compute_output_shape(self, input_shape):
        return [self.num_tasks] + input_shape

def custom_init(shape, dtype=None):
    x = tf.constant([[0.9, 0.1], [0.1, 0.9]])
    return x

if __name__ == "__main__":
    inputs = tf.keras.layers.Input(shape=[27, 107, 50])
    num_tasks = 2
    tops = [inputs] * num_tasks
    for task_id in range(num_tasks):
        in_tensor = tops[task_id]
        conv = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(10, 1),
        )(in_tensor)
        tops[task_id] = conv

    cs = CrossStitch(num_tasks)(tops)
    tops = tf.unstack(cs, axis=0)
    model = tf.keras.Model(inputs=inputs, outputs=tops)
    print(model.trainable_weights)
    model.summary()
