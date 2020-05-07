import tensorflow as tf

def generate_text(model, char2idx, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = start_string

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()

    tf.random.set_seed(1)
    for i in range(num_generate):
        if i > 200:
            print(str(i) + text_generated[-1])
        predictions = model(input_eval)
        # remove the batch
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the
        predictions = predictions / temperature
        predicted_id = 0
        while predicted_id == 0:
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append([predicted_id])
    return text_generated