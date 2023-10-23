class OpenVinoEngine:
    def __init__(self, path):
        from openvino.runtime import Core, Layout, get_batch  # noqa
        core = Core()
        xml_path = path
        bin_path = path.replace('.xml', '.bin')
        ov_model = core.read_model(model=xml_path, weights=bin_path)
        if ov_model.get_parameters()[0].get_layout().empty:
            ov_model.get_parameters()[0].set_layout(Layout('NCHW'))
        self.compiled_model = core.compile_model(ov_model, device_name='AUTO')  # AUTO selects best available device

    def forward(self, x):
        out = self.compiled_model(x)
        return out
