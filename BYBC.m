clc; clear; close all;

pkg load io  % Para leer archivos Excel en Octave

% === Cargar datos de entrenamiento desde un solo Excel ===
datos_train = xlsread('SVMTBC1.xlsx');
X_train = datos_train(:, 1:end-1);
y_train = datos_train(:, end);

% === Cargar datos de prueba desde dos Excel separados ===
benigno_test = xlsread('SVMPBC0.xlsx');
maligno_test = xlsread('SVMPBC1.xlsx');

X_test = [benigno_test(:, 1:end-1); maligno_test(:, 1:end-1)];
y_test = [-1 * ones(size(benigno_test, 1), 1); ones(size(maligno_test, 1), 1)];

% === Normalización robusta (mediana e IQR) ===
med_train = median(X_train);
iqr_train = iqr(X_train);
iqr_train(iqr_train == 0) = 1e-6; % evitar división por cero

X_train_norm = (X_train - med_train) ./ iqr_train;
X_test_norm = (X_test - med_train) ./ iqr_train;

% === Entrenamiento ===
classes = unique(y_train);
num_classes = length(classes);
n = size(X_train_norm, 2);

P_class = zeros(num_classes, 1);
mu = zeros(num_classes, n);
sigma = zeros(num_classes, n);

for i = 1:num_classes
    X_c = X_train_norm(y_train == classes(i), :);
    mu(i, :) = mean(X_c);
    sigma(i, :) = std(X_c);
    sigma(i, sigma(i,:) == 0) = 1e-6; % evitar división por cero
    P_class(i) = size(X_c, 1) / size(X_train_norm, 1);
end

% === Función de probabilidad gaussiana en log para evitar underflow ===
function y_pred = predict_nb_log(X_new, mu, sigma, P_class, classes)
    num_samples = size(X_new, 1);
    num_classes = length(P_class);
    y_pred = zeros(num_samples, 1);
    for i = 1:num_samples
        log_probs = zeros(num_classes, 1);
        for j = 1:num_classes
            log_p = -0.5*log(2*pi*sigma(j,:).^2) - ((X_new(i,:) - mu(j,:)).^2) ./ (2*sigma(j,:).^2);
            log_probs(j) = sum(log_p) + log(P_class(j));
        end
        [~, idx] = max(log_probs);
        y_pred(i) = classes(idx);
    end
end

% === Predicción con la función mejorada ===
y_pred = predict_nb_log(X_test_norm, mu, sigma, P_class, classes);

% === Cálculo de métricas ===
TP = sum((y_pred == 1) & (y_test == 1));
TN = sum((y_pred == -1) & (y_test == -1));
FP = sum((y_pred == 1) & (y_test == -1));
FN = sum((y_pred == -1) & (y_test == 1));

accuracy = (TP + TN) / length(y_test);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
F1 = 2 * (precision * recall) / (precision + recall);

% === Resultados ===
fprintf('\n=== Resultados Mejorados ===\n');
fprintf('Precisión general (accuracy): %.2f%%\n', accuracy * 100);
fprintf('Precisión (precision): %.2f%%\n', precision * 100);
fprintf('Sensibilidad (recall): %.2f%%\n', recall * 100);
fprintf('F1 Score: %.2f%%\n', F1 * 100);

% === Matriz de confusión ===
conf_matrix = [TN, FP; FN, TP];

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

% Etiquetas
title('Matriz de Confusión - Naive Bayes');
xlabel('Predicho');
ylabel('Real');
set(gca, 'XTick', [1 2], 'XTickLabel', {'-1', '1'});
set(gca, 'YTick', [1 2], 'YTickLabel', {'-1', '1'});

% Anotar los valores en las celdas
textStrings = num2str(conf_matrix(:));
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:2);
hStrings = text(x(:), y(:), textStrings(:), ...
                'HorizontalAlignment', 'center', 'Color', 'black', 'FontSize', 14);


% === Guardar modelo en .mat para la GUI ===
save('nb_model.mat', ...
     'med_train', 'iqr_train', ...
     'mu', 'sigma', 'P_class', 'classes');
disp('Modelo Naive Bayes guardado en nb_model.mat');

