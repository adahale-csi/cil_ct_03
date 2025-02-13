import numpy as np
import matplotlib.pyplot as plt
from cil.framework import AcquisitionGeometry
from cil.io import TIFFWriter
from cil.processors import TransmissionAbsorptionConverter
from cil.plugins.astra import FBP, ProjectionOperator

# Create a simple phantom
N = 128
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
phantom = ((X**2 + Y**2) <= 0.8**2).astype(float)

# Set up geometry
angles = np.linspace(0, np.pi, 180)
ag = AcquisitionGeometry.create_Parallel2D().set_angles(angles).set_panel(N)

# Create projections
PO = ProjectionOperator(ag)
sino = PO.direct(phantom)

# Convert to absorption data
converter = TransmissionAbsorptionConverter()
absorption_data = converter(sino)

# Perform FBP reconstruction
fbp = FBP(absorption_data)
reconstruction = fbp()

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(phantom, cmap='gray')
ax1.set_title('Original Phantom')
ax2.imshow(sino.array, cmap='gray', aspect='auto')
ax2.set_title('Sinogram')
ax3.imshow(reconstruction.array, cmap='gray')
ax3.set_title('FBP Reconstruction')
plt.tight_layout()
plt.show()

# Save the reconstructed image
writer = TIFFWriter(data=reconstruction, file_name='reconstruction.tiff')
writer.write()
