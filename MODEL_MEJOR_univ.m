% División de datos en conjuntos de entrenamiento, validación y prueba
ndto = 288*365;
datos=G_24h(1:ndto,8);
posiciones_cero=find(datos==0);
datos(posiciones_cero)=0.1;
datos_entrenamiento = datos(1:floor(0.75*ndto)); % 75% de los datos para entrenamiento
datos_validacion = datos(floor(0.75*ndto)+1:(floor(0.75*ndto)+1 + floor(0.15*ndto))); % 15% de los datos para validación
datos_prueba = datos((floor(0.75*ndto)+1 + floor(0.15*ndto))+1:end); % 10% de los datos para prueba

%normalización de los datos
mu=mean(datos_entrenamiento);
sigma=std(datos_entrenamiento);
datos_entrenamiento = (datos_entrenamiento-mu)/sigma;
datos_validacion = (datos_validacion-mu)/sigma;
datos_prueba = (datos_prueba-mu)/sigma;

% Creación de secuencias de datos de entrada y salida para entrenamiento
secuenciaSize = 6; % Tamaño de la ventana temporal D
horizonte = 1; % Horizonte de tiempo h
datos_entrenamiento_entrada = cell(1,length(datos_entrenamiento) - secuenciaSize - horizonte);
datos_entrenamiento_salida = [];
for i = 1:(length(datos_entrenamiento) - secuenciaSize - horizonte)
    datos_entrenamiento_entrada{i}= datos_entrenamiento(i+secuenciaSize-1:-1:i);
    datos_entrenamiento_salida = [datos_entrenamiento_salida; datos_entrenamiento(i+secuenciaSize-1+horizonte)];
end

% Creación de secuencias de datos de entrada y salida para validación
datos_validacion_entrada = cell(1,length(datos_validacion) - secuenciaSize - horizonte);
datos_validacion_salida = [];
for i = 1:(length(datos_validacion) - secuenciaSize - horizonte)
    datos_validacion_entrada{i} = datos_validacion(i+secuenciaSize-1:-1:i);
    datos_validacion_salida = [datos_validacion_salida; datos_validacion(i+secuenciaSize-1+horizonte)];
end

% Creación de secuencias de datos de entrada y salida para prueba
datos_prueba_entrada = cell(1,length(datos_prueba) - secuenciaSize - horizonte);
datos_prueba_salida = [];
for i = 1:(length(datos_prueba) - secuenciaSize -horizonte)
    datos_prueba_entrada{i} = datos_prueba(i+secuenciaSize-1:-1:i);
    datos_prueba_salida = [datos_prueba_salida; datos_prueba(i+secuenciaSize-1+horizonte)];
end

% Configuración de la red LSTM
num_features = secuenciaSize; % Número de características en los datos de entrada (en este caso, solo la potencia fotovoltaica)
num_hidden_units = 128; % Número de unidades ocultas en la capa LSTM
num_epochs = 200; % Número de épocas de entrenamiento
mini_batch_size = 128; % Tamaño del mini-lote
initial_learnRate = 0.005; %Tasa de aprendizaje inicial

% Creación de la arquitectura de la red LSTM
layers = [ ...
    sequenceInputLayer(num_features)
%     lstmLayer(num_hidden_units,'OutputMode','sequence')
%     dropoutLayer(0.5)
%     lstmLayer(64,'OutputMode','sequence')
%     dropoutLayer(0.5)
    bilstmLayer(num_hidden_units,'OutputMode','last')
    dropoutLayer(0.5)
    reluLayer
    fullyConnectedLayer(1)
    maeRegressionLayer('mae')];%regressionLayer]; 

% Opciones de entrenamiento
options = trainingOptions('adam', ...
    'MaxEpochs', num_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'ValidationData', {datos_validacion_entrada, datos_validacion_salida}, ... 
    'ValidationFrequency',50,...
    'InitialLearnRate',initial_learnRate,...
    'Verbose',0,...
    'Plots', 'training-progress');
   
    
% Entrenamiento del modelo LSTM
modelo = trainNetwork(datos_entrenamiento_entrada, datos_entrenamiento_salida, layers, options);

% Predicción de un paso adelante en los datos de prueba
predicciones_prueba = predict(modelo, datos_prueba_entrada');

% predicciones_prueba=zeros(1,length(datos_prueba_entrada));
% for i=1:length(datos_prueba_entrada)
% predicciones_prueba(i) = predict(modelo, datos_prueba_entrada(:,i));
% end

%valores escalados a originales
predicciones_prueba_originales=predicciones_prueba*sigma+mu;
datos_prueba_salida_originales=datos_prueba_salida*sigma+mu;

% Cálculo de métricas de rendimiento (por ejemplo, RMSE)
fprintf('************************************************\n ');
fprintf('Retrasos (D) en entradas: %.0f\n', secuenciaSize);
fprintf('Horizonte de tiempo (H): %.0f\n', horizonte);
fprintf('Unidades ocultas en capa LSTM (NU): %.0f\n', num_hidden_units);
fprintf('Tamaño del mini-lote (Batch): %.0f\n', mini_batch_size);
fprintf('Tasa de aprendizaje (learnrate): %.4f\n', initial_learnRate);
fprintf('Epocas de entrenamiento (Epoc): %.0f\n', num_epochs);
rmse = sqrt(mean((datos_prueba_salida_originales - predicciones_prueba_originales).^2));
fprintf('Error cuadrático medio (RMSE(kW)) en datos de prueba: %.4f\n', rmse);

mae = mean(abs(datos_prueba_salida_originales - predicciones_prueba_originales));
fprintf('Error absoluto medio (MAE(kW)) en datos de prueba: %.4f\n', mae);

mape = mean(abs((datos_prueba_salida - predicciones_prueba)./datos_prueba_salida))*100;
fprintf('Porcentaje de error absoluto medio (MAPE) en datos de prueba: %.4f\n', mape);

r = corr(predicciones_prueba_originales,datos_prueba_salida_originales)*100;
fprintf('Coeficiente de Correlación (r) entre predición y objetivos: %.4f\n', r);
%% 

% Visualización de las predicciones y los datos reales
figure;
plot(datos_prueba_salida_originales);
hold on;
plot(predicciones_prueba_originales);
legend('Datos de Prueba', 'Predicciones');
xlabel('Tiempo');
ylabel('Potencia Fotovoltaica');
title('Predicción de un paso adelante con LSTM');

