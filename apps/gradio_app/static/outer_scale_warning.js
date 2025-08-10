(value) => {
    const warningElement = document.getElementById('warning-text');
    if (warningElement) {
        if (value > 4) {
            warningElement.innerHTML = '<span style="color:red">To ensure optimal output quality, please set the <code>Outer Scale</code> to a value of 4 or less. The suggested range is from 1 to 4.</span>';
        } else {
            warningElement.innerHTML = '';
        }
    }
    return value; // Return the value to maintain slider functionality
}