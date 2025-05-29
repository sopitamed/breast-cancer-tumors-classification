clc; close all; clear all;
pkg load io

% ------ Carga y normalización ------
Data = xlsread('SVMTBC1.xlsx');
X    = Data(:,1:30);
yAll = Data(:,31);

Xmin = min(X);
Xmax = max(X);
Xn   = (X - Xmin) ./ (Xmax - Xmin);

X0 = Xn(yAll==-1, :);
X1 = Xn(yAll==+1, :);

% ------ Selección de pivotes y entrenamiento SVM ------
[n0,~] = size(X0); [n1,~] = size(X1);

k = 2 ; % Número de pivotes por clase %2 91.04

% Calcular matriz de distancias entre clases
D = zeros(n0,n1);
for i = 1:n0
    for j = 1:n1
        D(i,j) = norm(X0(i,:) - X1(j,:));
    end
end

% Encontrar los k pares más cercanos sin repetir muestras
[D_sorted, idx_sorted] = sort(D(:));
pivots = zeros(2*k, 31); % k pivotes negativos y k positivos

count = 0;
used_neg = false(n0,1);
used_pos = false(n1,1);
for idx = 1:length(idx_sorted)
    [i0, j1] = ind2sub(size(D), idx_sorted(idx));
    if ~used_neg(i0) && ~used_pos(j1)
        count++;
        pivots(count,:) = [X0(i0,:), -1];
        count++;
        pivots(count,:) = [X1(j1,:), +1];
        used_neg(i0) = true;
        used_pos(j1) = true;
    end
    if count >= 2*k
        break;
    end
end

% Armar matriz M y vector rhs para resolver sistema lineal
M = zeros(2*k);
rhs = zeros(2*k,1);
for i = 1:2*k
    for j = 1:2*k
        M(i,j) = dot(pivots(i,1:30).*pivots(i,31), pivots(j,1:30).*pivots(j,31));
    end
    rhs(i) = pivots(i,31);
end

a = M \ rhs;

% Calcular vector de pesos w y sesgo b
w = zeros(1,30);
for i = 1:2*k
    w += a(i)*pivots(i,31)*pivots(i,1:30);
end

b_vals = pivots(:,31) - pivots(:,1:30)*w';
b = mean(b_vals);

disp(['=== SVM entrenado con pivotes seleccionados (k=' num2str(k) ' por clase) ===']);
disp('Pesos w:'); disp(w);
disp('Sesgo b:'); disp(b);

% ------ Evaluación con datos de prueba ------
Test_Pos = xlsread('SVMPBC1.xlsx');    % clase +1
Test_Neg = xlsread('SVMPBC0.xlsx');    % clase -1

Test_Pos = Test_Pos(:,1:30);
Test_Neg = Test_Neg(:,1:30);

% Normaliza pruebas con Xmin/Xmax del entrenamiento
Test_Pos_N = (Test_Pos - Xmin) ./ (Xmax - Xmin);
Test_Neg_N = (Test_Neg - Xmin) ./ (Xmax - Xmin);

TP = 0; FP = 0; TN = 0; FN = 0;

for i = 1:size(Test_Pos_N,1)
    x = Test_Pos_N(i,:);
    score = dot(w,x) + b;
    y_pred = sign(score);
    if y_pred == +1
        TP++;
    else
        FN++;
    end
end

for i = 1:size(Test_Neg_N,1)
    x = Test_Neg_N(i,:);
    score = dot(w,x) + b;
    y_pred = sign(score);
    if y_pred == -1
        TN++;
    else
        FP++;
    end
end

% === Visualización de la matriz de confusión ===
conf_matrix = [TN, FP; FN, TP];  % Real vs predicho

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
title('Matriz de Confusión - SVM con pivotes');
xlabel('Predicho');
ylabel('Real');
set(gca, 'XTick', [1 2], 'XTickLabel', {'-1', '1'});
set(gca, 'YTick', [1 2], 'YTickLabel', {'-1', '1'});

% Anotar los valores en cada celda
textStrings = num2str(conf_matrix(:));
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:2);
hStrings = text(x(:), y(:), textStrings(:), ...
                'HorizontalAlignment', 'center', 'Color', 'black', 'FontSize', 14);


if F1 > 91
    save('svm_model_F1gt92.mat','w','b','Xmin','Xmax');
end

% Función auxiliar ternaria (opcional)
function out = ternary(cond, tStr, fStr)
    if cond, out = tStr; else out = fStr; end
end

