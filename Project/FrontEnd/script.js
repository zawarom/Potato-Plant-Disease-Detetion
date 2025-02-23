document.getElementById("uploadForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    const resultDiv = document.getElementById("result");
    const uploadedImage = document.getElementById("uploadedImage");
    const predictionText = document.getElementById("prediction");
    const suggestionBox = document.getElementById("suggestion-box");
    const suggestionText = document.getElementById("suggestion-text");
    const suggestionLink = document.getElementById("suggestion-link");

    if (!file) {
        alert("Please upload an image.");
        return;
    }

    // Display uploaded image preview
    const reader = new FileReader();
    reader.onload = function (e) {
        uploadedImage.src = e.target.result;
        resultDiv.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Prepare FormData
    const formData = new FormData();
    formData.append("file", file);

    // Disable button while processing
    const button = document.querySelector("button");
    button.disabled = true;
    button.textContent = "Processing...";

    try {
        // Send request to FastAPI backend
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            predictionText.textContent = `Error: ${data.error}`;
        } else {
            predictionText.textContent = `Prediction: ${data.class} (Confidence: ${data.confidence.toFixed(2)}%)`;
            resultDiv.style.display = "block";

            // Show appropriate suggestions based on the prediction
            let diseaseInfo = {
                "Early Blight": {
                    text: "Your potato plant has Early Blight. Learn how to manage it.",
                    link: "https://www.apsnet.org/edcenter/disandpath/fungal/pdlessons/Pages/EarlyBlightPotato.aspx"
                },
                "Late Blight": {
                    text: "Your potato plant has Late Blight. Find treatment options here.",
                    link: "https://www.apsnet.org/edcenter/disandpath/oomycete/pdlessons/Pages/LateBlightPotato.aspx"
                },
                "Healthy": {
                    text: "Your potato plant is healthy! Keep up the good care.",
                    link: "https://cropaia.com/blog/guide-to-potato-cultivation/"
                }
            };

            if (data.class in diseaseInfo) {
                suggestionText.textContent = diseaseInfo[data.class].text;
                suggestionLink.href = diseaseInfo[data.class].link;
                suggestionBox.style.display = "block";
            } else {
                suggestionBox.style.display = "none";
            }
        }
    } catch (error) {
        predictionText.textContent = "Error connecting to the server.";
    }

    // Re-enable button
    button.disabled = false;
    button.textContent = "Classify Image";
});
