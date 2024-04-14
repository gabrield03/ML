document.addEventListener('DOMContentLoaded', function() {
    var submitButton = document.querySelector('button');
    var textarea = document.getElementById('freeform');

    submitButton.addEventListener('click', function() {
        var emailContent = textarea.value;

        // POST to backend
        fetch('/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email: emailContent })
        })
        .then(response => response.json())
        .then(data => {
            // Handle the response from the backend
            console.log(data);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
