clc; close all; clear all;
pkg load io;

global Xmin Xmax
Xmin = [6.981 9.71 43.79 143.5 0.06251 0.01938 0 0 0.1167 0.04996 ...
        0.1115 0.3602 0.757 6.802 0.001713 0.002252 0 0 0.007882 0.0008948 ...
        7.93 12.02 50.41 185.2 0.07117 0.02729 0 0 0.1655 0.05504];

Xmax = [28.11 39.28 188.5 2499 0.1447 0.3454 0.4268 0.2012 0.304 0.09744 ...
        2.873 4.885 21.98 525.6 0.03113 0.1354 0.396 0.05279 0.07895 0.02984 ...
        33.12 49.54 220.8 3432 0.2226 1.058 1.252 0.291 0.6638 0.2075];


function crear_gui_clasificacion()
  global Xmin Xmax
  feat = { 'radius_mean','texture_mean','perimeter_mean','area_mean', ...
           'smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean', ...
           'radius_se','texture_se','perimeter_se','area_se','smoothness_se', ...
           'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se', ...
           'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst', ...
           'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'};

  scrsz = get(0,'ScreenSize');
  winW = scrsz(3) - 200;
  winH = scrsz(4) - 100;

  f = figure('Name','Clasificador de Vectores','Position',[100,50,winW,winH], ...
             'Color',[1 0.95 0.97]);

  n = numel(feat);
  handles.edits = zeros(1,n);

  panel = uipanel(f,'Title','Ingrese 30 características', ...
                  'Position',[0.02 0.33 0.96 0.65], ...
                  'BackgroundColor',[1 1 1], ...
                  'FontSize', 11);

  cols = 5; rows = 6;
  ancho = 160; alto = 22;
  sepX = 180; sepY = 70;

  totalAnchoGrid = cols * sepX;
  panelW = winW * 0.96;
  baseX = (panelW - totalAnchoGrid) / 2;
  baseY = 400;

  for idx = 1:n
    c = mod(idx-1, cols) + 1;
    r = floor((idx-1)/cols) + 1;

    x0 = baseX + (c-1)*sepX;
    y0 = baseY - (r-1)*sepY;

    % Label de la característica
    uicontrol(panel, ...
        'Style','text', ...
        'Position',[x0, y0, ancho, 20], ...
        'String', feat{idx}, ...
        'FontSize',8, ...
        'BackgroundColor',[1 1 1], ...
        'HorizontalAlignment','left');

    % Campo de entrada
    handles.edits(idx) = uicontrol(panel, ...
       'Style','edit', ...
       'Position',[x0, y0-22, ancho, alto], ...
       'BackgroundColor',[1 1 1], ...
       'ForegroundColor',[0 0 0], ...
       'String', '', ...
       'Tag', sprintf('e%d',idx));

    % Texto con rango permitido debajo del edit
    rango_str = sprintf('Range: %.3f - %.3f', Xmin(idx), Xmax(idx));
    uicontrol(panel, ...
        'Style','text', ...
        'Position',[x0, y0-45, ancho, 15], ...
        'String', rango_str, ...
        'FontSize',7, ...
        'ForegroundColor',[0.4 0.4 0.4], ...
        'BackgroundColor',[1 1 1], ...
        'HorizontalAlignment','left');
  end

  % Botón "Clasificar"
  uicontrol(f, ...
    'Style','pushbutton', ...
    'String','Clasificar', ...
    'FontSize', 11, ...
    'BackgroundColor', [1 0.6 0.8], ...
    'ForegroundColor', [0 0 0], ...
    'Position', [(winW - 100)/2, 70, 100, 40], ...
    'Callback', @(src,event) callback_clasificar(guidata(src)));

  % Botón "Limpiar campos"
  uicontrol(f, ...
    'Style','pushbutton', ...
    'String','Limpiar campos', ...
    'FontSize', 10, ...
    'BackgroundColor', [0.8 0.8 0.8], ...
    'ForegroundColor', [0 0 0], ...
    'Position', [(winW - 100)/2 + 120, 70, 100, 40], ...
    'Callback', @(~,~) limpiar_todos_los_campos(handles));

  ancho_resultado = 600;
  centroX = (winW - ancho_resultado) / 2;

  handles.res_svm = uicontrol(f, 'Style', 'text', ...
      'Position', [centroX, 180, ancho_resultado, 25], ...
      'String', 'SVM: ', ...
      'FontSize', 15, ...
      'ForegroundColor', [0.5 0 0.25], ...
      'BackgroundColor', [1 0.95 0.97], ...
      'HorizontalAlignment', 'left');

  handles.res_nn = uicontrol(f, 'Style', 'text', ...
      'Position', [centroX, 150, ancho_resultado, 25], ...
      'String', 'Red Neuronal: ', ...
      'FontSize', 15, ...
      'ForegroundColor', [0.5 0 0.25], ...
      'BackgroundColor', [1 0.95 0.97], ...
      'HorizontalAlignment', 'left');

  handles.res_nb = uicontrol(f, 'Style', 'text', ...
      'Position', [centroX, 120, ancho_resultado, 25], ...
      'String', 'Naive Bayes: ', ...
      'FontSize', 15, ...
      'ForegroundColor', [0.5 0 0.25], ...
      'BackgroundColor', [1 0.95 0.97], ...
      'HorizontalAlignment', 'left');

  guidata(f, handles);
end

function limpiar_todos_los_campos(handles)
  for i = 1:numel(handles.edits)
    set(handles.edits(i), 'String', '');
    set(handles.edits(i), 'ForegroundColor', [0 0 0]); % Texto negro
  end
end

function limpiar_placeholder(src)
  ejemplo = get(src, 'UserData');
  val = get(src, 'String');
  if strcmp(val, ejemplo)
      set(src, 'String', '');
      set(src, 'ForegroundColor', [0 0 0]);
  end
end


function callback_clasificar(handles)
  n = numel(handles.edits);
  x = zeros(1,n);
  for i = 1:n
    ejemplo = get(handles.edits(i), 'UserData');
    val = get(handles.edits(i), 'String');
    if strcmp(val, ejemplo) || isempty(val)
      errordlg(sprintf('Debe ingresar un valor para la característica #%d', i));
      return;
    end
    num = str2double(val);
    if isnan(num)
      errordlg(sprintf('Valor inválido en la característica #%d', i));
      return;
    end
    x(i) = num;
  end

  res_svm = clasificar_svm(x);
  res_nn = clasificar_nn(x);
  res_nb = clasificar_nb(x);

  set(handles.res_svm, 'String', sprintf('SVM: %s', resultado_texto(res_svm)));
  set(handles.res_nn, 'String', sprintf('Red Neuronal: %s', resultado_texto(res_nn)));
  set(handles.res_nb, 'String', sprintf('Naive Bayes: %s', resultado_texto(res_nb)));
end

function txt = resultado_texto(clase)
  if isequal(clase, 1)
    txt = 'Maligno';
  elseif isequal(clase, -1) || isequal(clase, 0)
    txt = 'Benigno';
  else
    txt = sprintf('Clase desconocida: %d', clase);
  end
end

function clase = clasificar_svm(x)
  data = load('svm_model_F1gt92.mat', 'w', 'b', 'Xmin', 'Xmax');
  w = data.w; b = data.b;
  Xmin = data.Xmin; Xmax = data.Xmax;
  xn = (x - Xmin) ./ (Xmax - Xmin);
  score = w * xn' + b;
  clase = sign(score);
end

function clase = clasificar_nn(x)
  data = load('red_cancer_mama_opt.mat', 'best_model', 'mu', 'sigma');
  bm = data.best_model; mu = data.mu; sigma = data.sigma;
  xn = (x - mu) ./ sigma; xn = xn';
  Z1 = bm.W1 * xn + bm.b1; A1 = max(0, Z1);
  Z2 = bm.W2 * A1 + bm.b2;
  expZ = exp(Z2 - max(Z2)); A2 = expZ ./ sum(expZ);
  [~, idx] = max(A2); clase = idx - 1;
end

function y_pred = clasificar_nb(x)
  data = load('nb_model.mat', 'med_train','iqr_train','mu','sigma','P_class','classes');
  med = data.med_train; iqr_ = data.iqr_train;
  mu = data.mu; sigma = data.sigma;
  P_cls = data.P_class; cls = data.classes;
  xn = (x - med) ./ iqr_;
  num_c = length(P_cls); log_probs = zeros(num_c,1);
  for j = 1:num_c
    log_p = -0.5*log(2*pi*sigma(j,:).^2) - ((xn - mu(j,:)).^2)./(2*sigma(j,:).^2);
    log_probs(j) = sum(log_p) + log(P_cls(j));
  end
  [~, idx] = max(log_probs); y_pred = cls(idx);
end

crear_gui_clasificacion();

