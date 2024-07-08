// function changeButton() {
//     var button = document.getElementById('myButton');
//     button.classList.remove('btn-success');
//     button.classList.add('btn-warning');
//     button.textContent = 'Detener';
// }



// function toggleButton() {
//     var button = document.getElementById('myButton');
//     if (button.classList.contains('btn-success')) {  
//     // Para cuando se quiera detener el programa
//         button.classList.remove('btn-success');
//         button.classList.add('btn-danger');
//         button.textContent = 'Detener';
//     } 
//     else {
//     // Para cuando se quiera inicializar el programa
//         button.classList.remove('btn-danger');
//         button.classList.add('btn-success');
//         button.textContent = 'Reproducir';
//     }
// }

function toggleButton() {
    var button = document.getElementById('myButton');
    if (button.classList.contains('btn-success')) {
        // Para cuando se quiera inicializar el programa
        fetch('/start_camera', {method: 'POST'})
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                button.classList.remove('btn-success');
                button.classList.add('btn-danger');
                button.textContent = 'Detener';
            }
        });
    } else {
        // Para cuando se quiera detener el programa
        fetch('/stop_camera', {method: 'POST'})
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                button.classList.remove('btn-danger');
                button.classList.add('btn-success');
                button.textContent = 'Reproducir';
            }
        });
    }
}