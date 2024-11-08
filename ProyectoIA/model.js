// Esperamos a que la página cargue
window.onload = async () => {
    try {
        // Cargar el modelo de Teachable Machine
        const model = await tf.loadLayersModel('modeloIA/model.json');
        console.log("Modelo cargado exitosamente!");
        
        // Ahora que el modelo está cargado, podemos agregarle la funcionalidad de predicción
        setupCameraAndPrediction(model);
    } catch (error) {
        console.error("Error al cargar el modelo:", error);
    }
};

// Función que se encargará de la captura de imagen y predicción
async function setupCameraAndPrediction(model) {
    const video = document.getElementById('cameraStream');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');

    // Cuando se haga clic en el botón de capturar imagen
    document.getElementById("captureButton").addEventListener("click", async () => {
        // Establecer el tamaño del canvas según el video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Dibujar el video en el canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convertir la imagen en tensor para poder procesarla
        let img = tf.browser.fromPixels(canvas);

        // Redimensionar la imagen y normalizarla para la predicción (ajusta según el tamaño que espera tu modelo)
        img = tf.image.resizeBilinear(img, [224, 224]); // Ajusta según el tamaño de entrada de tu modelo
        img = img.div(255.0); // Normaliza entre 0 y 1

        // Realizar la predicción
        const prediction = await model.predict(img.expandDims(0)); // expandDims para añadir dimensión de batch

        // Aquí puedes procesar la predicción, por ejemplo:
        const predictionArray = prediction.dataSync();
        const classId = predictionArray.indexOf(Math.max(...predictionArray));

        // Mostrar la clase predecida
        console.log(`Predicción: Clase ${classId}`);
    });
}
