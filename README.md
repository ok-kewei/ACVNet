# ACVNet
## Acknowledgments

This project is based on ACVNet (https://github.com/gangweiX/ACVNet)
I have modified this work for my own personal use, and the modifications. All credit for the original idea and code goes to the original author.

## Input & Output of the Code
### Input Images  

<table>
  <tr>
    <td><b>Left Image</b></td>
    <td><b>Right Image</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/e1a4cc69-4c28-4c17-9587-5e619323b521" width="400" height="300"></td>
    <td><img src="https://github.com/user-attachments/assets/3b53e9ae-f20c-4f5a-b297-e71ecc555208" width="400" height="300"></td>
  </tr>
</table>

### Output: Depth Map  
<img src="https://github.com/user-attachments/assets/6a5b1264-f4d9-4bf8-94fb-eaabe0e3e5db" width="800" height="600">  
</br>When you hover over the output image, it will display depth information corresponding to the pixel of the image.
---

## Running the Code  

To run the test and generate the depth map, execute:  

```bash
python test.py

```
## Note 
1. Input dimensions have be divisible by 16 due to the architecture's downsampling layers.
2. Error message encountered: "return torch.batch_norm(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 600.00 MiB. GPU 0 has a total capacity of 23.65 GiB of which 197.31 MiB is free. Process 6973 has 382.00 MiB memory in use"." </br>
To fix or mitigate the issue:<br>
   1. Reduce maxdisp <br>
      parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')<br>
   2. Reduce test_batch_size<br>
      parser.add_argument('--test_batch_size', type=int, default=1)<br>
   3. Reduce num_workers in DataLoader<br>
      TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)<br>
