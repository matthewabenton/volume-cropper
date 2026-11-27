// Bio-Formats' calibration of microscope files (read-only), for comparison
// with readers_0_15_check.py. Headless Fiji does not print to the terminal on
// Windows, so it writes a text file. Usage (Windows; Linux/Mac: export, and
// ':' between the files):
//   set BF_FILES=<folder>\a.lif;<folder>\b.czi
//   set BF_OUT=<scratch folder>\bf_sizes.txt
//   <Fiji>\fiji-windows-x64.exe --headless --run readers_0_15_bioformats.groovy
import loci.formats.ImageReader
import loci.common.services.ServiceFactory
import loci.formats.services.OMEXMLService

def files = System.getenv("BF_FILES").split(java.io.File.pathSeparator)
def out = new File(System.getenv("BF_OUT"))
def lines = []
def service = new ServiceFactory().getInstance(OMEXMLService)
try {
    for (f in files) {
        def meta = service.createOMEXMLMetadata()
        def reader = new ImageReader()
        reader.setMetadataStore(meta)
        reader.setId(f)
        lines << ("FILE " + f + " series " + reader.getSeriesCount())
        for (s in 0..<reader.getSeriesCount()) {
            reader.setSeries(s)
            def px = meta.getPixelsPhysicalSizeX(s)
            def py = meta.getPixelsPhysicalSizeY(s)
            def pz = meta.getPixelsPhysicalSizeZ(s)
            def ti = meta.getPixelsTimeIncrement(s)
            lines << ("  series " + s + " " + meta.getImageName(s)
                + " XYZCT " + reader.getSizeX() + "x" + reader.getSizeY() + "x"
                + reader.getSizeZ() + "x" + reader.getSizeC() + "x" + reader.getSizeT()
                + " | X " + (px == null ? "?" : px.value() + " " + px.unit().getSymbol())
                + " | Y " + (py == null ? "?" : py.value() + " " + py.unit().getSymbol())
                + " | Z " + (pz == null ? "?" : pz.value() + " " + pz.unit().getSymbol())
                + " | T " + (ti == null ? "?" : ti.value() + " " + ti.unit().getSymbol()))
            if (reader.getSizeT() > 2) {
                def times = []
                for (p in 0..<Math.min(meta.getPlaneCount(s), 5)) {
                    def dt = meta.getPlaneDeltaT(s, p)
                    times << (dt == null ? "?" : dt.value())
                }
                lines << ("    first plane times: " + times)
            }
        }
        reader.close()
    }
    lines << "DONE"
} catch (Throwable e) {
    lines << ("ERROR " + e)
}
out.text = lines.join(System.lineSeparator()) + System.lineSeparator()
