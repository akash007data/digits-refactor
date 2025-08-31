from sklearn.model_selection import train_test_split
from utils import (
    load_digits_data,
    visualize_training_samples,
    prepare_data,
    train_model,
    predict_digits,
    visualize_predictions,
    print_reports,
    rebuild_report_from_confusion_matrix,
)


def main():
    digits = load_digits_data()
    visualize_training_samples(digits)
    data = prepare_data(digits)

    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    clf = train_model(X_train, y_train)
    predicted = predict_digits(clf, X_test)
    visualize_predictions(X_test, predicted)

    disp = print_reports(clf, y_test, predicted)
    rebuild_report_from_confusion_matrix(disp)


if __name__ == "__main__":
    main()
