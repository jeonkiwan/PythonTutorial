
import vtk


def volumerendering_core():
	
	reader = vtk.vtkMetaImageReader()
	reader.SetFileName('segmentation_final.mhd')
	reader.Update()

	imag = reader.GetOutput()

	mapper = vtk.vtkGPUVolumeRayCastMapper()
	mapper.SetInputConnection(reader.GetOutputPort())

	# mapper.SetInputConnection(imag.GetProducePort())
	mapper.SetBlendModeToComposite()

	color = vtk.vtkColorTransferFunction()
	color.AddHSVPoint( -910,  0.00, 0.00, 0.00)
	color.AddHSVPoint( -650,  0.10, 0.10, 0.80)
	color.AddHSVPoint( -295,  0.00, 0.80, 0.65)
	color.AddHSVPoint(  360,  0.12, 0.10, 0.80)
	color.AddHSVPoint(  800,  0.00, 0.00, 0.00)

	opacity = vtk.vtkPiecewiseFunction()
	opacity.AddPoint(-1099, 0.00)
	opacity.AddPoint( -900, 0.005)
	opacity.AddPoint( -675, 0.00)
	opacity.AddPoint( -320, 0.00)
	opacity.AddPoint(  -15, 0.35)
	opacity.AddPoint(  215, 0.05)
	opacity.AddPoint(  530, 0.00)
	
	v_property = vtk.vtkVolumeProperty()
	v_property.SetColor(color)
	v_property.SetScalarOpacity(opacity)
	v_property.SetInterpolationTypeToLinear()
	v_property.ShadeOn()
	v_property.SetAmbient(0.4)
	v_property.SetDiffuse(0.6)
	v_property.SetSpecular(0.2)

	volume = vtk.vtkVolume()
	volume.SetMapper(mapper)
	volume.SetProperty(v_property)

	ren = vtk.vtkRenderer()
	win = vtk.vtkRenderWindow()
	win.AddRenderer(ren)
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(win)

	interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
	iren.SetInteractorStyle(interactorStyle)

	ren.AddViewProp(volume)

	camera =  ren.GetActiveCamera()
	c = volume.GetCenter()
	camera.SetFocalPoint(c[0], c[1], c[2])
	camera.SetPosition(c[0], c[1]-800, c[2])
	camera.SetViewUp(0, 0, 1)

	# Increase the size of the render window
	win.SetSize(800, 800)

	# Interact with the data.
	iren.Initialize()
	win.Render()
	iren.Start()

if __name__=='__main__':
	volumerendering_core()