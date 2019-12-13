# FE 595 Final

## Anand Patel, Dave Debreceni, Francesco Fabozzi

This website allows a user to upload a file with data to perform either classification (via K-Nearest Neighbors or a Classification Tree) or regression (via Regression Tree). The number of neighbors and maximum tree depth are modifiable by the user when uploading for KNN and CART respectively.

Once the data is uploaded and the model is specified, the results will be shown with a graphic that varies depending on which model was chosen:

* **For K-Nearest Neighbors**: A scatter plot with the first two features on either axis, and the color of each point indicates the class it was assigned to. Below the graph will show the overall accuracy rate on the testing data.

* **For a Decision Tree**: The graph will show the actual testing data as well as the predicted values according to the fitted model. The user will be able to see visually how well the model performed.


**NOTE**: To correctly install the libraries required for this project, please run `pip install -r requirements.txt --upgrade` before running the Flask app. Because the official release for the `mpld3` library has a bug, the `requirements.txt` file points to a correct branch in the official repository with the bug fixed.
