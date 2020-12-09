import os
import numpy as np
from copy import deepcopy as dcopy
import warnings
from ..logging import log, warn, debug, error
from .lagrit_helper import *


class Faceset:
    """
    Object that stores faceset formatting data
    """

    def __init__(self, fs_type, data, metadata=None):

        if metadata is None:
            metadata = {}

        self._has_type = fs_type
        self._data = data
        self._metadata = metadata


def FacesetFromElevations(heights: list, keep_body: bool = False) -> Faceset:
    """
    Facesets are generated and return from the heights array:
    a faceset will be created in the layers defined by `heights`.

    For example, with the `heights` array

    ```python
    heights = [50.,100.,150.,200.]
    ```

    five facesets will be created:

    * all surface elements lower than 50 meters
    * all surface elements between 50 and 100 meters
    * all surface elements between 100 and 150 meters
    * all surface elements between 150 and 200 meters
    * all surface elements greater than 200 meters

    Another approach is to split the elevation range into layers using
    properties `min_z` and `max_z`:

    ```python
    >>> print('Elevation range = ({0},{1})'.format(dem.min_z,dem.max_z))
    Elevation range = (2365.3,3942.2)

    >>> heights = np.linspace(dem.min_z,dem.max_z,10)
    >>> print(heights)
    array([2365.3       , 2540.51111111, 2715.72222222, 2890.93333333,
           3066.14444444, 3241.35555556, 3416.56666667, 3591.77777778,
           3766.98888889, 3942.2       ])

    ```

    # Arguments
    dem_object (tinerator.DEM): an instance of tinerator.DEM class
    heights (list<float>): a list of vertical (z) layers
    keep_body (bool): when True, elevation-based facesets are applied across
    the *entire mesh*. When False, elevation-based facesets only apply to
    the top layer.

    # Returns
    A Faceset object
    """
    return Faceset(
        "__FROM_ELEVATION", heights, metadata={"keep_body": keep_body}
    )


def FacesetFromSides(coords: np.ndarray, top_layer: bool = False) -> Faceset:
    """
    Operates on side facesets *only*.

    Constructs discretized side facesets based on the coords array.
    `coords` should contain one [x,y] pairs at each point a new sideset
    should be defined. Further, these points must be ordered clockwise.

    For an example, consider a square that spans 0 to 1 in both the
    x and y planes. The top, right, and bottom facesets are represented
    in the drawing below:

    ```
        1
     _______
    |       |
    |       | 2
    |       |
     -------
        3

    ```

    To construct `coords` properly, the array would look like:

    ```
    [0.,1.], # top
    [1.,1.], # right
    [1.,0.]  # bottom
    ```

    Note the points are ordered clock-wise.

    ------------------------

    By default, these sidesets will be applied to all layers.
    We can apply these sidesets to only the top layer (to capture an
    outlet, for example) by using flag `top_layer=True`.

    # Arguments
    coords (np.ndarray): clockwise array of points indicating faceset junctions
    top_layer (bool): when True, apply to only the top layer. when False, apply
    to all layers.

    # Returns
    A Faceset class instance
    """

    at_layers = None
    if top_layer:
        at_layers = [0]

    # Pre-process at_layers data
    if at_layers is None:
        at_layers = [-1]
    elif isinstance(at_layers, int):
        at_layers = [at_layers]

    # Verify data integrity
    assert isinstance(
        at_layers, (list, np.ndarray)
    ), "at_layers must be a list"
    assert all(
        isinstance(x, int) for x in at_layers
    ), "at_layers values must be int"

    return Faceset("__SIDESETS", coords, metadata={"layers": at_layers})


def FacesetBasic(
    has_top: bool = True, has_sides: bool = True, has_bottom: bool = True
) -> Faceset:
    """
    Generates basic facesets. Using the flags, you can define one or
    more of:

    * Top faceset
    * Side faceset
    * Bottom faceset

    # Arguments
    has_top (bool): generate a top faceset
    has_sides (bool): generate a sides faceset
    has_bottom (bool): generate a bottom faceset

    # Returns
    A Faceset class instance
    """
    return Faceset(
        "__NAIVE",
        None,
        metadata={"top": has_top, "sides": has_sides, "bottom": has_bottom},
    )


def __facesets_from_coordinates(coords: dict, boundary: np.ndarray):
    """
    Converts coordinates (used for sideset generation) into arrays
    pointing to boundary node numbers.

    That is, converts clockwise coordinates into a single line mesh
    attribute.

    When written to a file, this can be used for side set generation.
    """
    # Currently, only handles 'top' and 'all'.
    from scipy.spatial import distance

    facesets = {}

    for key in coords:
        mat_ids = np.full((np.shape(boundary)[0],), 1, dtype=int)
        fs = []

        # Iterate over given coordinates and find the closest boundary point...
        for c in coords[key]:
            ind = distance.cdist([c], boundary[:, :2]).argmin()
            fs.append(ind)

        # TODO: Band-aid fix.
        if len(fs) == 2:
            # Reverse order of fs, only so that the
            # notion of 'clockwise ordering matters'
            # stays constant.
            fs = fs[::-1]
        else:
            fs.sort(reverse=True)

        # Map the interim space as a new faceset.
        # 'Unmarked' facesets have a default filled value of 1
        for i in range(len(fs)):

            if fs[-1] > fs[i]:
                mat_ids[fs[-1] :] = i + 2
                mat_ids[: fs[i]] = i + 2
            else:
                mat_ids[fs[-1] : fs[i]] = i + 2

        facesets[key] = mat_ids

    return facesets


def __driver_naive(lg, surface_mesh, top, bottom, sides):
    """
    Simple driver that constructs one or more of:

    * Top faceset
    * Bottom faceset
    * Side faceset

    """

    faceset_fnames = []

    mo_surf = surface_mesh.copy()

    for att in ["itetclr0", "itetclr1", "facecol", "idface0", "idelem0"]:
        mo_surf.delatt(att)

    mo_surf.select()
    mo_surf.setatt("itetclr", 3)

    mo_surf.dump("surface_mesh_test.inp")

    ptop = mo_surf.pset_attribute(
        "layertyp", -2, comparison="eq", stride=[1, 0, 0]
    )

    pbot = mo_surf.pset_attribute(
        "layertyp", -1, comparison="eq", stride=[1, 0, 0]
    )

    etop = ptop.eltset(membership="exclusive")
    ebot = pbot.eltset(membership="exclusive")

    mo_surf.setatt("itetclr", 100, stride=["eltset", "get", etop.name])
    mo_surf.setatt("itetclr", 200, stride=["eltset", "get", ebot.name])

    esides = mo_surf.eltset_attribute("itetclr", 50, boolstr="lt")

    if top:
        log("Generating top faceset")
        mo_tmp = mo_surf.copy()
        edel = mo_tmp.eltset_not([etop])
        mo_tmp.rmpoint_eltset(edel, resetpts_itp=False)

        fname = "fs_naive_top.avs"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lg.sendline(
                "dump / avs2 / " + fname + "/" + mo_tmp.name + "/ 0 0 0 2"
            )

        faceset_fnames.append(fname)
        mo_tmp.delete()

    if bottom:
        log("Generating bottom faceset")
        mo_tmp = mo_surf.copy()
        edel = mo_tmp.eltset_not([ebot])
        mo_tmp.rmpoint_eltset(edel, resetpts_itp=False)

        fname = "fs_naive_bottom.avs"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lg.sendline(
                "dump / avs2 / " + fname + "/" + mo_tmp.name + "/ 0 0 0 2"
            )

        # if cfg.DEBUG_MODE:
        #    mo_tmp.dump("DEBUG_naive_bottom_fs.inp")

        faceset_fnames.append(fname)
        mo_tmp.delete()

    if sides:
        log("Generating sides faceset")
        mo_tmp = mo_surf.copy()
        edel = mo_tmp.eltset_not([esides])
        mo_tmp.rmpoint_eltset(edel, resetpts_itp=False)

        fname = "fs_naive_sides.avs"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lg.sendline(
                "dump / avs2 / " + fname + "/" + mo_tmp.name + "/ 0 0 0 2"
            )

        faceset_fnames.append(fname)
        mo_tmp.delete()

    mo_surf.delete()
    return faceset_fnames


def __driver_sidesets(lg, surface_mesh, has_top, boundary_file, full_sidesets):
    """
    A new technique for generating Exodus facesets from the material ID of
    boundary line segments.

    By providing the array `boundary_attributes`, of equal length to `boundary`,
    the line segment material IDs are used as indentifiers for unique facesets.

    The top and bottom facesets will always be generated; there will be a
    minimum of one side facesets if boundary_attributes is set to a uniform
    value, or up to length(boundary) number of facesets, if each value in
    boundary_attributes is unique.

    LaGriT methodology developed by Terry Ann Miller, Los Alamos Natl. Lab.
    """

    mo_surf = surface_mesh.copy()

    mo_surf.addatt("id_side", vtype="vint", rank="scalar", length="nelements")
    mo_surf.settets_normal()
    mo_surf.copyatt("itetclr", attname_sink="id_side", mo_src=mo_surf)

    for att in [
        "itetclr0",
        "idnode0",
        "idelem0",
        "facecol",
        "itetclr1",
        "idface0",
        "nlayers",
        "nnperlayer",
        "neperlayer",
        "ikey_utr",
    ]:
        mo_surf.delatt(att)

    # TODO: Why is this generating a warning?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmo_bndry = lg.read(boundary_file)

    cmo_bndry.resetpts_itp()

    # use stack attribute layertyp to set top and bottom
    # set all sides to default 3 all
    mo_surf.select()
    mo_surf.setatt("id_side", 3)

    ptop = mo_surf.pset_attribute(
        "layertyp", -2, comparison="eq", stride=[1, 0, 0]
    )
    pbot = mo_surf.pset_attribute(
        "layertyp", -1, comparison="eq", stride=[1, 0, 0]
    )

    etop = ptop.eltset(membership="exclusive")
    ebot = pbot.eltset(membership="exclusive")

    mo_surf.setatt("id_side", 2, stride=["eltset", "get", etop.name])
    mo_surf.setatt("id_side", 1, stride=["eltset", "get", ebot.name])

    mo_surf.copyatt("id_side", attname_sink="itetclr", mo_src=mo_surf)

    # Set default node imt based on top, bottom, sides
    # NOTE nodes at top/side edge are set to 3 side
    # change order of setatt to overwrite differently

    mo_surf.select()
    mo_surf.setatt("imt", 1)
    esides = mo_surf.eltset_attribute("id_side", 3)
    psides = esides.pset()

    mo_surf.setatt("imt", 2, stride=["pset", "get", ptop.name])
    mo_surf.setatt("imt", 1, stride=["pset", "get", pbot.name])
    mo_surf.setatt("imt", 3, stride=["pset", "get", psides.name])

    mo_surf.select()
    mo_surf.addatt("zsave", vtype="vdouble", rank="scalar", length="nnodes")
    mo_surf.copyatt("zic", mo_src=mo_surf, attname_sink="zsave")
    mo_surf.setatt("zic", 0.0)
    cmo_bndry.setatt("zic", 0.0)

    # INTERPOLATE boundary faces to side faces
    # and set numbering so 1 and 2 are top and bottom

    cmo_bndry.math(
        "add",
        "itetclr",
        value=2,
        stride=[1, 0, 0],
        cmosrc=cmo_bndry,
        attsrc="itetclr",
    )
    cmo_bndry.math(
        "add",
        "imt1",
        value=2,
        stride=[1, 0, 0],
        cmosrc=cmo_bndry,
        attsrc="imt1",
    )
    mo_surf.interpolate(
        "map",
        "id_side",
        cmo_bndry,
        "itetclr",
        stride=["eltset", "get", esides.name],
        flag_option="nearest, imt1",
    )

    if has_top:
        mo_surf.addatt(
            "ioutlet", vtype="vint", rank="scalar", length="nelements"
        )
        mo_surf.addatt(
            "ilayer", vtype="vint", rank="scalar", length="nelements"
        )
        mo_surf.interpolate(
            "map",
            "ioutlet",
            cmo_bndry,
            "ioutlet",
            stride=["eltset", "get", esides.name],
        )

    if has_top > 1:
        mo_surf.addatt(
            "iinlet", vtype="vint", rank="scalar", length="nelements"
        )
        mo_surf.addatt(
            "ilayer", vtype="vint", rank="scalar", length="nelements"
        )
        mo_surf.interpolate(
            "map",
            "iinlet",
            cmo_bndry,
            "iinlet",
            stride=["eltset", "get", esides.name],
        )

    mo_surf.copyatt("zsave", mo_src=mo_surf, attname_sink="zic")
    mo_surf.delatt("zsave")

    mo_surf.setatt("id_side", 2, stride=["eltset", "get", etop.name])
    mo_surf.setatt("id_side", 1, stride=["eltset", "get", ebot.name])

    # check material numbers, must be greater than 0
    # id_side is now ready for faceset selections
    mo_surf.copyatt("id_side", attname_sink="itetclr", mo_src=mo_surf)

    _all_facesets = []
    faceset_count = np.size(np.unique(full_sidesets)) + 2

    # This creates a top-layer boundary faceset by setting the area defined in
    # ioutlet to a unique value *only* on the top layer.
    if has_top:

        mo_surf.select()
        mo_surf.setatt("ilayer", 0.0)

        elay_inc = ptop.eltset(membership="inclusive")
        mo_surf.setatt("ilayer", 1, stride=["eltset", "get", elay_inc.name])
        mo_surf.setatt(
            "ilayer", 0, stride=["eltset", "get", etop.name]
        )  # ????

        e1 = mo_surf.eltset_attribute("ilayer", 1, boolstr="eq")
        e2 = mo_surf.eltset_attribute("ioutlet", 2, boolstr="eq")

        faceset_count += 1

        e_out1 = mo_surf.eltset_inter([e1, e2])
        mo_surf.setatt(
            "id_side", faceset_count, stride=["eltset", "get", e_out1]
        )

        # If outlet is defined with inlet...
        if has_top > 1:
            mo_surf.select()
            mo_surf.setatt("ilayer", 0.0)

            elay_inc = ptop.eltset(membership="inclusive")
            mo_surf.setatt(
                "ilayer", 1, stride=["eltset", "get", elay_inc.name]
            )
            mo_surf.setatt(
                "ilayer", 0, stride=["eltset", "get", etop.name]
            )  # ????

            e1 = mo_surf.eltset_attribute("ilayer", 1, boolstr="eq")
            e2 = mo_surf.eltset_attribute("iinlet", 2, boolstr="eq")

            faceset_count += 1

            e_out1 = mo_surf.eltset_inter([e1, e2])
            mo_surf.setatt(
                "id_side", faceset_count, stride=["eltset", "get", e_out1]
            )

            mo_surf.delatt("iinlet")

        mo_surf.delatt("ioutlet")
        mo_surf.delatt("ilayer")

    # Capture and write all facesets
    for ss_id in range(3, faceset_count + 1):
        fname = "ss%d_fs.faceset" % ss_id

        mo_tmp = mo_surf.copy()
        mo_tmp.select()
        e_keep = mo_tmp.eltset_attribute("id_side", ss_id, boolstr="eq")
        e_delete = mo_tmp.eltset_bool(boolstr="not", eset_list=[e_keep])
        mo_tmp.rmpoint_eltset(e_delete, compress=True, resetpts_itp=False)

        mo_tmp.delatt("layertyp")
        mo_tmp.delatt("id_side")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lg.sendline(
                "dump / avs2 / " + fname + "/" + mo_tmp.name + "/ 0 0 0 2"
            )

        mo_tmp.delete()
        _all_facesets.append(fname)

    # Cleaup!
    cmo_bndry.delete()
    mo_surf.delete()

    return _all_facesets


def __driver_top(lg, surface_mesh, heights, keep_body):
    """
    Driver for generating discretized top facesets.
    Should not be called out of the main facesets generation
    function.

    Developed by Ilhan Ozgen (EESA, Lawrence Berkeley
    National Laboratory).
    """

    # Sort from lowest to highest
    heights = heights.copy()

    if isinstance(heights, (int, float)):
        heights = [heights]

    heights.sort()

    # cfg.log.info("Preparing surface mesh")

    # --- GET OUTSIDE SURFACE MESH --------------------------- #

    mo_surf = surface_mesh.copy()
    mo_surf.select()
    mo_surf.setatt("itetclr", 3)
    ptop = mo_surf.pset_attribute("layertyp", -2)
    etop = ptop.eltset(membership="exclusive")
    mo_surf.setatt("itetclr", 2, stride=["eltset", "get", etop.name])

    mo = surface_mesh.copy()
    mo.select()

    for att in ["itetclr0", "itetclr1", "facecol", "idface0", "idelem0"]:
        mo.delatt(att)

    eall = mo.eltset_attribute("itetclr", 0, boolstr="ge")
    edel = mo.eltset_not([eall, etop])
    mo.rmpoint_eltset(edel, compress=True, resetpts_itp=False)

    if keep_body:
        ptop = mo.pset_attribute(
            "layertyp", -2, comparison="eq", stride=[1, 0, 0]
        )
        etop = ptop.eltset(membership="exclusive")

        edel = mo.eltset_not([etop])
        mo.rmpoint_eltset(edel, resetpts_itp=False)

    # -------------------------------------------------------- #

    # cfg.log.info("Creating cut planes")

    planes = []

    # Create a plane at spanning each given height
    for z in heights:
        cpl = lg.surface_plane(
            [0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], ibtype="intrface"
        )
        planes.append(cpl)

    # ------------------------------------------- #
    # Create regions covering all elevations,
    # with layers based on the 'heights' array

    # cfg.log.info("Finding elements within cut planes")

    count = len(planes)
    regions = []

    # Capture everything below min point
    regions.append(mo.region_bool("le " + planes[0].name))

    # Capture all intermediary layers
    for i in range(count - 1):
        r = mo.region_bool(
            "gt " + planes[i].name + " and le " + planes[i + 1].name
        )
        regions.append(r)

    # Capture everything above max point
    regions.append(mo.region_bool("gt " + planes[-1].name))

    # ------------------------------------------- #
    # Set material ID of mesh to a value based on
    # the captured eltset above

    itetclrs = [(101 + i) for i in range(len(regions))]

    for (i, r) in enumerate(regions):
        es = mo.eltset_region(r)
        mo.setatt("itetclr", itetclrs[i], stride=["eltset", "get", es.name])

    # ------------------------------------------- #
    # Iterate through each eltset and dump its
    # elements as a faceset

    fs_files = []

    for (i, iclr) in enumerate(itetclrs):
        # cfg.log.info("Generating top faceset %d / %d" % (i + 1, len(itetclrs)))
        fs_name = "fs_elevations_%d.avs" % (iclr)

        cut = mo.copy()

        if not keep_body:
            ptop = cut.pset_attribute(
                "layertyp", -2, comparison="eq", stride=[1, 0, 0]
            )
            ebot = ptop.eltset(membership="exclusive")
            edel = cut.eltset_not([ebot])
            cut.rmpoint_eltset(edel, compress=True, resetpts_itp=False)

        edel = cut.eltset_attribute("itetclr", iclr, boolstr="ne")
        cut.rmpoint_eltset(edel, compress=True, resetpts_itp=False)

        # Write facesets in AVS-UCD format if in debug mode
        # if cfg.DEBUG_MODE:
        #    cut.dump(fs_name.replace("avs", "inp"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lg.sendline(
                "dump / avs2 / " + fs_name + "/" + cut.name + "/ 0 0 0 2"
            )

        # cfg.log.debug("Wrote faceset %s" % fs_name)

        fs_files.append(fs_name)
        cut.delete()

    return fs_files


def write_facesets(lg, dem_object, facesets):
    """
    Given a DEM object and a list of Faceset class objects,
    this generates the corresponding faceset files.

    This function should not be used by an end-user unless they want
    AVS2 faceset files unconnected to an Exodus mesh.

    # Arguments
    dem_object (tinerator.DEM): DEM class instance to operate from
    facesets (list<tinerator.Faceset>): list of Faceset objects describing
    generation steps

    # Returns
    A list containing generated faceset files
    """

    boundary_file = "_boundary_line_colors.inp"

    _cleanup = []
    heights = None
    naive = False

    dem_object.add_empty_attribute("layertyp", "node", fill_value=0.0)
    dem_object.set_attribute("layertyp", -2.0, at_layer=0)
    dem_object.set_attribute("layertyp", -1.0, at_layer=-1)

    if not isinstance(facesets, list):
        facesets = [facesets]

    # -- SIDESET PREPARATION -------------------------------------- #
    # cfg.log.info("Preparing sidesets...")
    # Prepare the sideset objects - get full sides && top layer
    sidesets = {}
    for fs in facesets:
        if fs._has_type == "__SIDESETS":
            md_layers = fs._metadata["layers"]
            if md_layers == [-1]:
                sidesets["all"] = fs._data
            elif md_layers == [0]:
                if "top" in sidesets:
                    sidesets["top2"] = fs._data
                else:
                    sidesets["top"] = fs._data
            else:
                raise ValueError("An unknown error occurred")
        elif fs._has_type == "__FROM_ELEVATION":
            heights = fs._data
            keep_body = fs._metadata["keep_body"]
        elif fs._has_type == "__NAIVE":
            naive = fs._metadata

    # If sidesets exist...
    # The DEM boundary represented as a line mesh is
    # required to get discretized sidesets.
    # If top layer sidesets exist as well, then this
    # is represented as a line mesh attribute
    if bool(sidesets):
        sidesets = __facesets_from_coordinates(sidesets, dem_object.boundary)

        cell_atts = None
        has_top = False

        if "all" in sidesets:
            full_sidesets = sidesets["all"]
        else:
            # TODO!
            raise ValueError("Undefined behaviour...")

        if "top" in sidesets:
            has_top = True
            cell_atts = {"ioutlet": sidesets["top"]}

        if "top2" in sidesets:
            has_top = 2
            cell_atts["iinlet"] = sidesets["top2"]

        full_sidesets = dcopy(full_sidesets) - np.min(full_sidesets) + 1

        # Test that the array does not have values that 'skip' an integer,
        # i.e., [1,4,3] instead of [1,3,2]
        # TODO: might be redundant now
        assert np.all(
            np.unique(full_sidesets)
            == np.array(range(1, np.size(np.unique(full_sidesets)) + 1))
        ), "full_sidesets cannot contain non-sequential values"

        # Generate the line mesh
        conn = line_connectivity(dem_object.boundary, connect_ends=True)
        write_line(
            dem_object.boundary,
            boundary_file,
            connections=conn,
            material_id=full_sidesets,
            cell_atts=cell_atts,
        )

    # dem_object._stacked_mesh = lg.read('__tmp_pri.inp')

    # -- MAIN PREPARATION -------------------------------------- #

    dem_object.save("_stacked_mesh.inp")
    _stacked_mesh = lg.read("_stacked_mesh.inp")
    _stacked_mesh.dump("stacked_test_hmm.inp")

    _cleanup.append(boundary_file)
    _cleanup.append("_stacked_mesh.inp")

    lg.sendline(f"cmo/select/{_stacked_mesh.name}")
    _stacked_mesh.resetpts_itp()

    lg.sendline("resetpts/itp")
    lg.sendline("createpts/median")
    lg.sendline(
        f"sort/{_stacked_mesh.name}/index/ascending/ikey/itetclr zmed ymed xmed"
    )
    lg.sendline("reorder/{0}/ikey".format(_stacked_mesh.name))
    lg.sendline("cmo/DELATT/{0}/ikey".format(_stacked_mesh.name))
    lg.sendline("cmo/DELATT/{0}/xmed".format(_stacked_mesh.name))
    lg.sendline("cmo/DELATT/{0}/ymed".format(_stacked_mesh.name))
    lg.sendline("cmo/DELATT/{0}/zmed".format(_stacked_mesh.name))
    lg.sendline("cmo/DELATT/{0}/ikey".format(_stacked_mesh.name))

    cmo_in = _stacked_mesh.copy()

    # Extract surface w/ cell & face attributes to get the outside face
    # to element relationships

    try:
        raise Exception("Unknown bug in standard surfmesh...move to catch")
        mo_surf = lg.extract_surfmesh(
            cmo_in=cmo_in, stride=[1, 0, 0], external=True, resetpts_itp=True
        )

    except Exception as e:
        # This is NOT a good solution
        # TODO: Bug fix help wanted!
        import subprocess

        # cfg.log.debug("Caught surface mesh error - trying shell")

        tmp_infile = "_EOF_ERROR_SURFMESH.lgi"
        mesh_prism = "_TEMP_MESH_PRI.inp"
        mesh_surf = "_TEMP_MESH_SURF.inp"

        cmo_in.dump(mesh_prism)

        with open(tmp_infile, "w") as f:
            f.write(
                Infiles._surf_mesh_backup(
                    mesh_prism, mesh_surf, skip_sort=True
                )
            )

        subprocess.check_output(
            lg.lagrit_exe + " < " + tmp_infile,
            stderr=subprocess.STDOUT,
            shell=True,
        )

        mo_surf = lg.read(mesh_surf)

        cleanup([tmp_infile, mesh_surf, mesh_prism])

    cmo_in.delete()
    mo_surf.select()

    exported_fs = []

    # Generate discretized sidesets
    if bool(sidesets):
        new_fs = __driver_sidesets(
            lg, mo_surf, has_top, boundary_file, full_sidesets
        )

        exported_fs.extend(new_fs)

    # Generate discretized surface facesets
    if heights is not None:
        new_fs = __driver_top(lg, mo_surf, heights, keep_body)
        exported_fs.extend(new_fs)

    # Generate basic top, side, and bottom sidesets
    if naive:
        new_fs = __driver_naive(
            lg, mo_surf, naive["top"], naive["bottom"], naive["sides"]
        )

        exported_fs.extend(new_fs)

    dem_object.rm_attribute("layertyp")

    return exported_fs
