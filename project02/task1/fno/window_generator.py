import torch

class WindowGenerator:
    def __init__(self, data, targets, input_width, output_width, shift, stride=1):
        self.data = data
        self.targets = targets
        self.input_width = input_width
        self.output_width = output_width
        self.shift = shift
        self.stride = stride

        self.total_window_size = input_width + (shift - 1) * stride
        self.input_slice = slice(0, input_width * stride, stride)
        self.input_indices = torch.arange(self.total_window_size)[self.input_slice]

        self.output_start = self.total_window_size - output_width
        self.outputs_slice = slice(self.output_start, None)
        self.output_indices = torch.arange(self.total_window_size)[self.outputs_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices.numpy()}',
            f'Label indices: {self.output_indices.numpy()}'
        ])

    def __len__(self):
        return len(self.data) - self.total_window_size + 1

    def __getitem__(self, idx):
        input_data = self.data[idx + self.input_indices]
        outputs = self.targets[idx + self.output_indices]
        return input_data, outputs
