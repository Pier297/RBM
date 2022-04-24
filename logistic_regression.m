
function model = logistic_regression(X_train, Y_train, X_test, Y_test, structure_hidden)
    model = fitcnet(X_train, Y_train, "LayerSizes", structure_hidden);

    trainingAccuracy = 1 - loss(model, X_train, Y_train);
    
    fprintf('Train accuracy = %.3f\n', trainingAccuracy);

    testAccuracy = 1 - loss(model, X_test, Y_test);
    
    fprintf('Test accuracy = %.3f\n', testAccuracy);
    
%     figure
%     confusionchart(Y_train, predict(model, X_train))
%     sgtitle('Train Confusion Matrix')
%     saveas(gcf, 'Results\train_confusion_chart.png')
%     
%     figure
%     confusionchart(Y_test, predict(model, X_test))
%     sgtitle('Test Confusion Matrix')
%     saveas(gcf, 'Results\test_confusion_chart.png')
end