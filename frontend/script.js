document.addEventListener('DOMContentLoaded', () => {
    // Get references to HTML elements for inputs and outputs
    const temperatureInput = document.getElementById('temperature');
    const humidityInput = document.getElementById('humidity');
    const soilPhInput = document.getElementById('soil_ph');
    const lightIntensityInput = document.getElementById('light_intensity');
    const npkNInput = document.getElementById('npk_n');
    const npkPInput = document.getElementById('npk_p');
    const npkKInput = document.getElementById('npk_k');
    const rainStatusSelect = document.getElementById('rain_status');

    const recommendCropsBtn = document.getElementById('recommendCropsBtn');
    const cropRecommendationOutput = document.getElementById('cropRecommendationOutput');
    const cropListSpan = document.getElementById('cropList');

    const imageUploadInput = document.getElementById('imageUpload');
    const uploadedImageElem = document.getElementById('uploadedImage');
    const analyzeImageBtn = document.getElementById('analyzeImageBtn');

    const predictionOutput = document.getElementById('predictionOutput');
    const diseaseNameSpan = document.getElementById('diseaseName');
    const confidenceSpan = document.getElementById('confidence');

    const segmentedImageCard = document.getElementById('segmentedImageCard');
    const segmentedImageElem = document.getElementById('segmentedImage');

    const precautionsOutput = document.getElementById('precautionsOutput');
    const precautionsTextSpan = document.getElementById('precautionsText');
    const medicinesOutput = document.getElementById('medicinesOutput');
    const medicinesTextSpan = document.getElementById('medicinesText');

    const analysisStatusDiv = document.getElementById('analysisStatus'); 

    const messageBox = document.getElementById('messageBox');
    const messageText = document.getElementById('messageText');

    let base64Image = null; // To store the base64 encoded image

    // --- Helper function to show custom messages ---
    function showMessage(message) {
        messageText.textContent = message;
        messageBox.style.display = 'block';
    }

    // --- Function to toggle loading indicator ---
    function toggleLoading(show) {
        analysisStatusDiv.style.display = show ? 'flex' : 'none'; 
    }

    // --- Function to resize and convert image to base64 ---
    function resizeAndEncodeImage(file, maxWidth, maxHeight, quality, callback) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                let width = img.width;
                let height = img.height;

                // Calculate new dimensions to fit within maxWidth/maxHeight
                if (width > height) {
                    if (width > maxWidth) {
                        height *= maxWidth / width;
                        width = maxWidth;
                    }
                } else {
                    if (height > maxHeight) {
                        width *= maxHeight / height;
                        height = maxHeight;
                    }
                }

                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                // Convert canvas content to base64 with specified quality
                const resizedBase64 = canvas.toDataURL('image/jpeg', quality);
                callback(resizedBase64);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    // --- Image Upload Event Listener ---
    imageUploadInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // Reset UI
            uploadedImageElem.src = '#';
            uploadedImageElem.style.display = 'none';
            predictionOutput.style.display = 'none';
            segmentedImageCard.style.display = 'none';
            precautionsOutput.style.display = 'none';
            medicinesOutput.style.display = 'none';
            
            // Resize and encode the image
            resizeAndEncodeImage(file, 1024, 1024, 0.8, (resizedBase64) => {
                uploadedImageElem.src = resizedBase64;
                uploadedImageElem.style.display = 'block';
                base64Image = resizedBase64; // Store the resized base64 image
            });
        } else {
            uploadedImageElem.src = '#';
            uploadedImageElem.style.display = 'none';
            base64Image = null;
        }
    });

    // --- Recommend Crops Button Event Listener ---
    recommendCropsBtn.addEventListener('click', async () => {
        toggleLoading(true);
        cropRecommendationOutput.style.display = 'block';
        cropListSpan.textContent = 'Fetching recommendations...';

        const environmentalData = {
            temperature: parseFloat(temperatureInput.value),
            humidity: parseFloat(humidityInput.value),
            soil_ph: parseFloat(soilPhInput.value),
            light_intensity: parseFloat(lightIntensityInput.value),
            npk_n: parseFloat(npkNInput.value),
            npk_p: parseFloat(npkPInput.value),
            npk_k: parseFloat(npkKInput.value),
            rain_status: rainStatusSelect.value
        };

        try {
            const response = await fetch('/recommend_crops', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(environmentalData),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const crops = await response.json();
            if (crops && crops.length > 0 && crops[0].englishName !== "API Key Missing") {
                const formattedCrops = crops.map(crop => `${crop.englishName} (${crop.bengaliName})`).join('<br>- ');
                cropListSpan.innerHTML = `- ${formattedCrops}`;
            } else {
                cropListSpan.textContent = 'Could not recommend crops. Please check backend logs or API key.';
                showMessage('Failed to get crop recommendations. Please check backend server and API key.');
            }
        } catch (error) {
            console.error('Error recommending crops:', error);
            cropListSpan.textContent = 'Error fetching recommendations.';
            showMessage(`Error fetching crop recommendations: ${error.message}`);
        } finally {
            toggleLoading(false);
        }
    });

    // --- Analyze Image Button Event Listener ---
    analyzeImageBtn.addEventListener('click', async () => {
        if (!base64Image) {
            showMessage('Please upload an image first.');
            return;
        }

        toggleLoading(true);
        
        // Reset previous outputs
        predictionOutput.style.display = 'none';
        segmentedImageCard.style.display = 'none';
        precautionsOutput.style.display = 'none';
        medicinesOutput.style.display = 'none';

        const environmentalData = {
            temperature: parseFloat(temperatureInput.value),
            humidity: parseFloat(humidityInput.value),
            soil_ph: parseFloat(soilPhInput.value),
            light_intensity: parseFloat(lightIntensityInput.value),
            npk_n: parseFloat(npkNInput.value),
            npk_p: parseFloat(npkPInput.value),
            npk_k: parseFloat(npkKInput.value),
            rain_status: rainStatusSelect.value,
            image: base64Image // This is now the resized/compressed base64 image
        };

        try {
            const response = await fetch('/analyze_plant', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(environmentalData),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.disease_name && result.disease_name !== "API Key Missing") {
                diseaseNameSpan.textContent = result.disease_name;
                confidenceSpan.textContent = result.confidence;
                predictionOutput.style.display = 'block';
            } else {
                diseaseNameSpan.textContent = 'Could not classify. Check backend logs.';
                confidenceSpan.textContent = 'N/A';
                predictionOutput.style.display = 'block';
                showMessage(`AI Classification failed: ${result.disease_name}. Please try again or check your API key.`);
            }

            if (result.segmented_image) {
                segmentedImageElem.src = `data:image/png;base64,${result.segmented_image}`;
                segmentedImageCard.style.display = 'block';
            } else {
                segmentedImageCard.style.display = 'none';
            }

            if (result.remedies) {
                if (result.remedies.Precautions && result.remedies.Precautions !== "API Key Missing") {
                    precautionsTextSpan.innerHTML = result.remedies.Precautions.replace(/\n/g, '<br>');
                    precautionsOutput.style.display = 'block';
                } else {
                    precautionsOutput.style.display = 'none';
                    showMessage('Precautions not found in AI response or API key missing.');
                }
                
                if (result.remedies.Medicines && result.remedies.Medicines !== "API Key Missing") {
                    medicinesTextSpan.innerHTML = result.remedies.Medicines.replace(/\n/g, '<br>');
                    medicinesOutput.style.display = 'block';
                } else {
                    medicinesOutput.style.display = 'none';
                    showMessage('Medicines not found in AI response or API key missing.');
                }
            } else {
                showMessage('No AI-generated recommendations found.');
            }

        } catch (error) {
            console.error('Error analyzing plant:', error);
            showMessage(`Error analyzing plant: ${error.message}`);
        } finally {
            toggleLoading(false);
        }
    });
});
