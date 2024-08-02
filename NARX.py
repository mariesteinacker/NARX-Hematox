import numpy as np
import tensorflow as tf


class MinMax(tf.keras.constraints.Constraint):
    # Constraint to help set exponential smoothing within [0,1]
    def __init__(self, w_min, w_max):
        """
        Constraint to set parameters in a specified range [w_min, w_max]
        :param w_min: Minimum value
        :param w_max: Maximum value
        """
        super().__init__()
        self.w_min = w_min
        self.w_max = w_max

    def call(self, w):
        if tf.math.greater(w, self.w_max):
            return self.w_max
        else:
            return w * tf.cast(tf.math.greater_equal(w, self.w_min), w.dtype)


class ARX_RNN_ES(tf.keras.Model):
    """
    generalized ARX-RNN class for a given RNN cell type,
    single hidden RNN layer, based on AR-model from tutorial
    https://www.tensorflow.org/tutorials/structured_data/time_series
    """
    def __init__(self, cell, lags, inp=tf.keras.layers.Identity(),
                 out=tf.keras.layers.Dense(1)):
        """
        build an ARX-RNN model from a specified RNN cell type
        :param cell: RNN cell type
        :param lags: warm up size
        :param inp: input layer, identity or gaussian noise for training
        :param out: output layer
        """
        super().__init__()
        self.lags = lags
        self.cell = cell
        self.inp = inp
        # Also wrap the Cell in an RNN to simplify the `warmup` method.
        self.rnn = tf.keras.layers.RNN(self.cell, return_state=True)
        self.dense = out        # MISO model
        # exponential smoothing factor
        self.alpha = self.add_weight(name='ES', shape=(1,1),
                    initializer=tf.keras.initializers.Constant(value=1.0),
                    constraint=MinMax(w_min=0., w_max=1.),
                    trainable=True)

    def warmup(self, inputs):
        """This method returns a single time-step prediction and the internal
         state of the RNN:"""
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, rnn_units)
        x = self.inp(inputs)
        x, *state = self.rnn(x)
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None, mask=None):
        """
        iterate through time series via AR method
        """
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the RNN state.
        # returns single time step prediction based on warmup
        prediction, state = self.warmup(inputs[:,:self.lags])
        new_inputs = tf.concat([inputs[:, self.lags, 0].reshape(-1,1),
                                prediction], axis=-1)
        # Insert the first prediction.
        predictions.append(prediction)
        es_pred = prediction

        # saveguard that alpha is in [0,1]
        if self.alpha < 0.:
            self.alpha.assign(tf.zeros_like(self.alpha))
        elif self.alpha > 1.:
            self.alpha.assign(tf.ones_like(self.alpha))

        # Run the rest of the prediction steps.
        for n in range(1,inputs.shape[1]-self.lags):
            # Use the last prediction as input.
            # concat with exogenous input
            x = self.inp(new_inputs)
            # Execute one rnn step.
            x, state = self.cell(x, states=state,
                                      training=training)
            # Convert the rnn output to a prediction.
            prediction = self.dense(x)
            new_inputs = tf.concat([inputs[:, n + self.lags, 0].reshape(-1,1),
                                    prediction], axis=-1)
            # execute the exponential smoothing step
            es_pred = self.alpha * prediction + (1 - self.alpha) * es_pred
            # Add the prediction to the output.
            predictions.append(es_pred)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


class ARX_net_lagged(tf.keras.Model):
    """
    generalized ARX-FNN class for a given FNN or MLP,
    based on AR-model from tutorial
    https://www.tensorflow.org/tutorials/structured_data/time_series
    """
    def __init__(self, mlp, xlag, ylag, es=False):
        """
        build an ARX-FNN model from a specified MLP
        :param mlp: inner MLP
        :param xlag: memory for exogenous input
        :param ylag: memory for regressed input
        :param es: exponential smoothing, boolean
        """
        super().__init__()
        self.xlag = xlag
        self.ylag = ylag
        self.mlp = mlp
        self.alpha = self.add_weight(name='ES', shape=(1,1),
                    initializer=tf.keras.initializers.Constant(value=1.0),
                    constraint=MinMax(w_min=0., w_max=1.),
                    trainable=es)

    def call(self, inputs, training=None, mask=None):
        """
        iterate through time series via AR method
        :param inputs: inputs to iterate over, should be shaped such that
        last dimension corresponds to stacked ylag, xlag
        """
        xlag = self.xlag
        ylag = self.ylag

        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # returns single time step prediction based on warmup (first time step)
        prediction = self.mlp(inputs[:, :1])

        # these need to be seperated to iterate over the exogenous input,
        # but include the regressed output
        reg_input = tf.concat([inputs[:, :1, 1:ylag + 1], prediction],
                              axis=-1)
        exo_input = inputs[:, 1:2, ylag + 1:]
        new_inputs = tf.concat([reg_input, exo_input],
                               axis=-1)
        # Insert the first prediction.
        predictions.append(prediction)

        # predefine for ar step
        es_pred = prediction

        # saveguard for expontential smoothing
        if self.alpha < 0.:
            self.alpha.assign(tf.zeros_like(self.alpha))
        elif self.alpha > 1.:
            self.alpha.assign(tf.ones_like(self.alpha))

        # Run the rest of the prediction steps.
        for n in range(1, inputs.shape[1] - 1):
            # Use the last prediction as input.
            # concat with exogenous input
            x = new_inputs
            # Execute one step.
            prediction = self.mlp(x)

            # make next lagged input
            reg_input = tf.concat([reg_input[:, :1, 1:], prediction], axis=-1)
            exo_input = inputs[:, n + 1:n + 2, ylag + 1:]
            new_inputs = tf.concat([reg_input, exo_input], axis=-1)

            # Add the prediction to the output.
            # add exponential smoothing, if es=False then alpha=1
            es_pred = self.alpha * prediction + (1 - self.alpha) * es_pred
            predictions.append(es_pred)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # remove doubled dimension
        predictions = tf.squeeze(predictions, [-1])
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions