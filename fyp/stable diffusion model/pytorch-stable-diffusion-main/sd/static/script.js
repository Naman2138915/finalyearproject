const image = document.getElementById('image');
const cropper = new Cropper(image, {
    aspectRatio: 1,
});

document.querySelector('#btn-crop').addEventListener('click', function() {
    var croppedImage = cropper.getCroppedCanvas().toDataURL("image/png");
    document.getElementById('output').src = croppedImage;
    document.querySelector(".cropped-container").style.display = 'flex';
});

