def interpret_prediction(prediction):

    if prediction == 1:

        return (
            'High Risk of Default',
            'The customer is more likely '
            'to miss the next credit payment.'
        )

    return (
        'Low Risk of Default',
        'The customer is less likely '
        'to default on the next payment.'
    )