clc; close all; clear all;

pkg load io  % Si usas Octave, para leer xls

% ==============================
% 1. Carga y normalización de datos de entrenamiento desde un solo Excel
% ==============================

datos = xlsread('SVMTBC1.xlsx');  % Datos combinados
y = datos(:, end)';
y(y == -1) = 0;  % Convertir benigno (-1) a 0 para softmax

mu = mean(datos(:, 1:end-1));
sigma = std(datos(:, 1:end-1));
Dato_norm = (datos(:, 1:end-1) - mu) ./ sigma;
input = Dato_norm';

% ==============================
% 1.1 Carga y normalización de datos de prueba desde dos Excel separados
% ==============================

benigno_test = xlsread('SVMPBC0.xlsx');
maligno_test = xlsread('SVMPBC1.xlsx');

benigno_test_norm = (benigno_test(:, 1:end-1) - mu) ./ sigma;
maligno_test_norm = (maligno_test(:, 1:end-1) - mu) ./ sigma;

input_test = [benigno_test_norm; maligno_test_norm]';
y_test = [zeros(1, size(benigno_test_norm,1)), ones(1, size(maligno_test_norm,1))];

% ==============================
% 2. Hiperparámetros y funciones de activación
% ==============================

num_inputs = size(input, 1);
num_outputs = 2;
learning_rate = 0.01;
epochs = 4000;

hidden_layers = [10, 15, 20, 25];
lambdas = [0.01, 0.05, 0.1];

relu = @(Z) max(0, Z);
relu_derivative = @(Z) Z > 0;
softmax = @(Z) exp(Z - max(Z, [], 1)) ./ sum(exp(Z - max(Z, [], 1)), 1);

best_f1 = 0;

% ==============================
% 3. Búsqueda y entrenamiento
% ==============================

for num_hidden = hidden_layers
  for lambda = lambdas

    W1 = randn(num_hidden, num_inputs);
    W2 = randn(num_outputs, num_hidden);
    b1 = randn(num_hidden, 1) * 0.1;
    b2 = randn(num_outputs, 1) * 0.1;

    for epoch = 1:epochs
      Z1 = W1 * input + b1;
      A1 = relu(Z1);
      Z2 = W2 * A1 + b2;
      A2 = softmax(Z2);

      dZ2 = A2 - [1 - y; y];
      dW2 = dZ2 * A1';
      db2 = sum(dZ2, 2);

      dA1 = W2' * dZ2;
      dZ1 = dA1 .* relu_derivative(Z1);
      dW1 = dZ1 * input';
      db1 = sum(dZ1, 2);

      W1 = W1 - learning_rate * (dW1 + lambda * W1);
      W2 = W2 - learning_rate * (dW2 + lambda * W2);
      b1 = b1 - learning_rate * db1;
      b2 = b2 - learning_rate * db2;
    end

    % Evaluación
    Z1_test = W1 * input_test + b1;
    A1_test = relu(Z1_test);
    Z2_test = W2 * A1_test + b2;
    A2_test = softmax(Z2_test);
    [~, y_pred_test] = max(A2_test, [], 1);
    y_pred_test = y_pred_test - 1;

    TP = sum((y_pred_test == 1) & (y_test == 1));
    TN = sum((y_pred_test == 0) & (y_test == 0));
    FP = sum((y_pred_test == 1) & (y_test == 0));
    FN = sum((y_pred_test == 0) & (y_test == 1));

    confMat = [TP FP; FN TN];
    accuracy = (TP + TN) / sum(confMat(:)) * 100;
    precision = TP / (TP + FP) * 100;
    recall = TP / (TP + FN) * 100;
    F1 = 2 * (precision * recall) / (precision + recall);

    fprintf('Neur: %d, Lambda: %.3f, Accuracy: %.2f%%, Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%\n', ...
      num_hidden, lambda, accuracy, precision, recall, F1);

    if F1 > best_f1
      best_f1 = F1;
      best_model.W1 = W1;
      best_model.W2 = W2;
      best_model.b1 = b1;
      best_model.b2 = b2;
      best_model.F1 = F1;
      best_model.num_hidden = num_hidden;
      best_model.lambda = lambda;
      best_model.confMat = confMat;
      best_model.accuracy = accuracy;
      best_model.precision = precision;
      best_model.recall = recall;
    end

  end
end

% ==============================
% 4. Resultados finales y guardado
% ==============================

fprintf('\nMejor modelo con %d neuronas y lambda=%.3f:\n', best_model.num_hidden, best_model.lambda);
fprintf('Exactitud: %.2f%%\n', best_model.accuracy);
fprintf('Precision: %.2f%%\n', best_model.precision);
fprintf('Recall (sensibilidad): %.2f%%\n', best_model.recall);
fprintf('F1-score: %.2f%%\n', best_model.F1);
% === Visualización de la matriz de confusión ===
conf_matrix = [best_model.confMat(2,2), best_model.confMat(1,2);  % TN, FP
               best_model.confMat(2,1), best_model.confMat(1,1)]; % FN, TP

figure;
imagesc(conf_matrix);

% Definir colormap personalizado
color1 = [0.698, 0.933, 0.933];  % #b2eeee
color2 = [0.435, 0.780, 0.780];  % #6fc7c7
custom_map = [linspace(color1(1), color2(1), 64)', ...
              linspace(color1(2), color2(2), 64)', ...
              linspace(color1(3), color2(3), 64)'];
colormap(custom_map);
colorbar;

% Etiquetas de los ejes
title('Matriz de Confusión - Red Neuronal Softmax');
xlabel('Predicho');
ylabel('Real');
set(gca, 'XTick', [1 2], 'XTickLabel', {'0 (Benigno)', '1 (Maligno)'});
set(gca, 'YTick', [1 2], 'YTickLabel', {'0 (Benigno)', '1 (Maligno)'});

% Anotar los valores en cada celda
textStrings = num2str(conf_matrix(:));
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:2);
hStrings = text(x(:), y(:), textStrings(:), ...
                'HorizontalAlignment', 'center', 'Color', 'black', 'FontSize', 14);




if best_model.F1 > 98
  save('red_cancer_mama_opt.mat', 'best_model','mu','sigma');
  disp('Mejor modelo guardado correctamente.');
else
  disp('No se alcanzó el umbral para guardar el modelo.');
end

