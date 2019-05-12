import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import vtk

wdbc = pd.read_csv("./wdbc.csv")


# maldf = wdbc[wdbc.diagnosis == 'M']
# drop_indices = np.random.choice(maldf.index, int(0.75*len(maldf)), replace=False)
# wdbc = wdbc.drop(drop_indices)

data_column = wdbc.columns[2:]
X = wdbc[data_column].values
Y = pd.Categorical(wdbc.diagnosis).codes

train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size=0.2,
                                                    random_state=3333,
                                                    stratify=Y)


scaler = StandardScaler()
scaler.fit(train_X)

scaled_train_X = scaler.transform(train_X)
scaled_test_X = scaler.transform(test_X)

def vtk_show(renderer, width=800, height=800):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()
     
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()
     
    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = bytes(memoryview(writer.GetResult()))
    
    return Image(data)


def plot_embedding_3d(X, Y, title=None):

    renderer = vtk.vtkRenderer()
    renderer.GradientBackgroundOn();
    renderer.SetBackground2(0.05, 0.05, 0.05)
    renderer.SetBackground(0.25, 0.25, 0.31)

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    camera = renderer.GetActiveCamera()

    for i, pos in enumerate(X):
#         print(i, pos)
        text = vtk.vtkVectorText()
        text.SetText('M' if Y[i] > 0 else 'B')
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(text.GetOutputPort())
        actor = vtk.vtkFollower()
        actor.SetMapper(mapper)
        actor.SetScale(0.02, 0.02, 0.02)
        actor.AddPosition(pos)
        actor.SetCamera(camera)
        if Y[i]:
            actor.GetProperty().SetColor(1.0, 0.8, 0.25)
        else:
            actor.GetProperty().SetColor(0.25, 0.75, 0.9)    
        renderer.AddActor(actor)

    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()

    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(renderer)
    renwin.SetSize(800, 800)
    renwin.Render()

    
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(renwin)
    iren.Start()

#     focal = camera.GetFocalPoint()
#     pos = camera.GetPosition()
    
#     camera.SetFocalPoint(focal)
#     camera.SetPosition(pos)
    # camera.Azimuth(20);
    # camera.Elevation(40);
    # # renderer.SetActiveCamera(camera)
    # camera.Roll(50);
    
    # renderer.ResetCamera()
    # renderer.ResetCameraClippingRange()
        
    return renderer

# pca = PCA(n_components=3).fit(train_X)
# X_pca = pca.transform(train_X)

pca = PCA(n_components=3).fit(scaled_train_X)
X_pca = pca.transform(scaled_train_X)

plot_embedding_3d(X_pca, train_y, "Principal Components projection")
