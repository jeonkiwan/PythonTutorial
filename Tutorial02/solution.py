fig, ax = plt.subplots(figsize=(6, 6))    
buffer = sitk.GetArrayFromImage(image)

# eliminate axis 
ax.imshow(np.squeeze(buffer, axis=0), vmin=-240, vmax=160)
# or
# ax.imshow(buffer[0, :, :], vmin=-240, vmax=160)

ax.set_axis_off()