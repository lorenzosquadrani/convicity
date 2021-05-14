import matplotlib.pyplot as plt

def view_weights(model, rows, cols, figsize = None, save = False, path = None):
    
    if model.out_channels < rows*cols:
        print("Invalid combination of rows and columns!")
        return 0
    if save and (path is None):
        print("You need to specify a path to save the images!")
        return 0 
    
    if figsize is None:
        figsize = (cols*2.5,rows*2.5)
        
    fig, ax = plt.subplots(rows,cols, figsize= figsize)
    for i in range(rows):
        for j in range(cols):
            ax[i,j].imshow(model.weights[2*i + j , :].reshape(model.kernel_size, model.kernel_size))

    plt.show()




def view_weights(weights,out_channels,kernel_size,rows, cols, figsize = None, save = False, path = None):
    
    if out_channels < rows*cols:
        print("Invalid combination of rows and columns!")
        return 0
    if save and (path is None):
        print("You need to specify a path to save the images!")
        return 0 
    
    if figsize is None:
        figsize = (cols*2.5,rows*2.5)
        
    fig, ax = plt.subplots(rows,cols, figsize= figsize)
    for i in range(rows):
        for j in range(cols):
            ax[i,j].imshow(weights[2*i + j , :].reshape(kernel_size, kernel_size))

    plt.show()