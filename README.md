Deep Learning & AdS/CFT
=======================
Implementation of DL method from (Hashimoto, et al, 2018).

Deep Neural Network Representation of the Scalar Field in AdS Spacetime
-----------------------
For a scalar field theory in a <img src="https://render.githubusercontent.com/render/math?math=(d %2B 1)">-dimensional curved spacetime is written as

<img src="https://render.githubusercontent.com/render/math?math=S = \int d^{d %2B 1}x \sqrt{-\mathrm{det}g} \left[ -\frac{1}{2}(\partial_\mu\phi)^2 - \frac{1}{2}m^2\phi^2 - V(\phi) \right]">.

Suppose the field configuration depends only on <img src="https://render.githubusercontent.com/render/math?math=\eta">, which is the holographic direction. Then, the generic metric is given by

<img src="https://render.githubusercontent.com/render/math?math=ds^2 = -f(\eta)dt^2 %2B d\eta^2 %2B g(\eta) (dx_1^2 %2B \cdots %2B dx_{d-1}^2)">,

with the asymptotic AdS boundary condition <img src="https://render.githubusercontent.com/render/math?math=f \approx g \approx \mathrm{exp}[2\eta/L] (\eta\approx\infty)"> with the AdS radius <img src="https://render.githubusercontent.com/render/math?math=L">,and another boundary condition at the black hole horizon, <img src="https://render.githubusercontent.com/render/math?math=f \approx \eta^2, g \approx \mathrm{const.} (\eta\approx 0)">.

The classical equation of motion for the scalar field <img src="https://render.githubusercontent.com/render/math?math=\phi(\eta)"> is

<img src="https://render.githubusercontent.com/render/math?math=\partial_\eta \pi %2B h(\eta)\pi - m^2 \phi - \frac{\delta V[\phi]}{\delta \phi} = 0">,

where <img src="https://render.githubusercontent.com/render/math?math=\pi \equiv \partial_\eta \phi"> and <img src="https://render.githubusercontent.com/render/math?math=h(\eta) \equiv \partial_\eta \log \sqrt{f(\eta)g(\eta)^{d-1}}">.

To represent this equation of motion as a deep neural network, it can be discretized in the radial <img src="https://render.githubusercontent.com/render/math?math=\eta"> direction as the following

<img src="https://render.githubusercontent.com/render/math?math=\phi(\eta %2B \Delta \eta)=\phi(\eta) %2B \Delta\eta\pi(\eta)">,

<img src="https://render.githubusercontent.com/render/math?math=\pi(\eta %2B \Delta \eta)=\pi(\eta) - \Delta \eta \left( h(\eta)\pi(\eta) - m^2\phi(\eta) - \frac{\delta V(\phi)}{\delta \phi(\eta)} \right)">.

Input vector for the neural network will be <img src="https://render.githubusercontent.com/render/math?math=[x_1,x_2]^T = [\eta(\infty),\pi(\infty)]^T">, and it will propagate along the neural network, up to the black hole horizon at <img src="https://render.githubusercontent.com/render/math?math=\eta=0">. Each layer will be a fully connected layer with 2 input and output features. Weight matrix for n-th layer corresponds to the discretized equation of motion is

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+W%5E%7B%28n%29%7D+%3D+%5Cleft%5B+%5Cbegin%7Bmatrix%7D%0A1+%26+%5CDelta+%5Ceta+%5C%5C%0A%5CDelta+%5Ceta+m%5E2+%26+1+-+%5CDelta+%5Ceta+h%28%5Ceta%5E%7B%28n%29%7D%29%0A%5Cend%7Bmatrix%7D+%5Cright%5D%0A" 
alt="W^{(n)} = \left[ \begin{matrix}
1 & \Delta \eta \\
\Delta \eta m^2 & 1 - \Delta \eta h(\eta^{(n)})
\end{matrix} \right]
">,

and the activation function for each layer is

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cvarphi%28x_1%29+%26%3D+x_1%5C%2C%2C%5C%5C%0A%5Cvarphi%28x_2%29+%26%3D+x_2+%2B+%5CDelta+%5Ceta+%5Cfrac%7B%5Cdelta+V%28x_1%29%7D%7B%5Cdelta+x_1%7D%5C%2C.%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
\varphi(x_1) &= x_1\,,\\
\varphi(x_2) &= x_2 + \Delta \eta \frac{\delta V(x_1)}{\delta x_1}\,.
\end{align*}
">

The output layer has 2 input features and 1 output feature. Exact form of the output layer will be explained in the next section.

Application on AdS Schwartzchild Black Hole
--------------------------------

References
----------------------
[1] K. Hashimoto, S. Sugishita, A. Tanaka, and A. Tomiya. (2018). Deep Learning and the AdS/CFT Correspondence. _Phys. Rev. D_ 98 (2018) 4. 046019.
