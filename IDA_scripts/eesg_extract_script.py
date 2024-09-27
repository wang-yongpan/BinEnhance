# -*- encoding: utf-8 -*-

import json
import sys, os

from settings import *
sys.path.insert(0, PACKAGE_PATH)
sys.setrecursionlimit(3000)
import idc
import idautils
import idaapi
import networkx as nx

ida_ver = IDA_VERSION
if ida_ver >= 7.0:
    import ida_name
    import ida_bytes
    import ida_auto
    import ida_pro
    import ida_nalt
    GetSegName = idc.get_segm_name
    GetSegByName = idc.selector_by_name
    GetSegBySel = idc.get_segm_by_sel
    GetSegEnd = idc.get_segm_end
    isData = ida_bytes.is_data
    GetFlag = ida_bytes.get_full_flags
    isString = ida_bytes.is_strlit
    GetString = idc.get_strlit_contents
    GetName = ida_name.get_ea_name
    GetFuncName = idc.get_func_name
    AllFunctions = idautils.Functions
    CodeRefsTo = idautils.CodeRefsTo
    XrefsTo = idautils.XrefsTo
    Wait = ida_auto.auto_wait
    Exit = ida_pro.qexit
    GetInputFilePath = ida_nalt.get_input_file_path
    GetFuncAttr = idc.get_func_attr
    FlowChart = idaapi.FlowChart
    Get_func = idaapi.get_func
    Get_ItemSize = idc.get_item_size
    Get_Byte = idc.get_wide_byte
    Get_Word = idc.get_wide_word
    Get_Dword = idc.get_wide_dword
    Get_Qword = idc.get_qword
    GetBytes = idc.get_bytes
    pass
else:
    GetSegName = idc.SegName
    GetSegByName = idc.SegByName
    GetSegBySel = idc.SegByBase
    GetSegEnd = idc.SegEnd
    isData = idc.isData
    GetFlag = idc.GetFlags
    isString = idc.isASCII
    GetString = idc.GetString
    GetName = idc.GetTrueName
    GetFuncName = idc.GetFunctionName
    AllFunctions = idautils.Functions
    CodeRefsTo = idautils.CodeRefsTo
    XrefsTo = idautils.XrefsTo
    Wait = idc.Wait
    Exit = idc.Exit
    GetInputFilePath = idc.GetInputFilePath
    GetFuncAttr = idc.GetFunctionAttr
    FlowChart = idaapi.FlowChart
    Get_func = idaapi.get_func
    Get_ItemSize = idc.ItemSize
    Get_Byte = idc.Byte
    Get_Word = idc.Word
    Get_Dword = idc.Dword
    Get_Qword = idc.Qword
    GeyBytes = idc.GetManyBytes
    pass

def is_func_in_plt(function_ea):
    segm_name = GetSegName(function_ea).lower()
    if "got" not in segm_name and "plt" not in segm_name:
        return False
    else:
        return True

def is_func_in_text(function_ea):
    segm_name = GetSegName(function_ea).lower()
    if "text" not in segm_name:
        return False
    else:
        return True

def is_func_in_extern(function_ea):
    segm_name = GetSegName(function_ea).lower()
    if "extern" not in segm_name:
        return False
    else:
        return True

def get_var_value(ea):
    d = []
    size = Get_ItemSize(ea)
    data = GetBytes(ea, size)
    for i in range(size):
        if data[i] == 0 or data[i] == 255:
            continue
        d.append(data[i])
    return d

def extract_global_var(data_segs, functions):
    var_dict = {}
    str_dict = {}
    for data_seg in data_segs:
        idata_seg_selector = GetSegByName(data_seg)
        idata_seg_startea = GetSegBySel(idata_seg_selector)
        idata_seg_endea = GetSegEnd(idata_seg_startea)
        for seg_ea in range(idata_seg_startea, idata_seg_endea):
            flags = GetFlag(seg_ea)
            if not isData(flags) or seg_ea in var_dict:
                continue
            if isString(flags):
                content = str(GetString(seg_ea).decode("utf-8"))
                # if len(content) < 5:
                #     continue
                tp = "string"
            else:
                # if the evaluation results is less than our results, you can delete the {value}, only use the var_name 
                # our method to process the {value} is too simple, because it is not our main focus
                var_name = GetName(seg_ea)
                value = get_var_value(seg_ea)
                if not check_is_auto_var(var_name):
                    if len(value) == 0:
                        content = str(var_name)
                    else:
                        content = str(value)
                else:
                    if len(value) == 0:
                        content = str(var_name) + "|||" + "auto_var"
                    else:
                        content = str(value)
                tp = "variable"
                # if len(content) == 3:
                #     continue
            func_xrefs = {}
            for xref in XrefsTo(seg_ea):
                func_name = str(get_unified_funcname(xref.frm))
                if len(func_name) == 0 or func_name == None:
                    continue
                if func_name not in func_xrefs and func_name in functions:
                    func_xrefs[func_name] = xref.frm
            if len(func_xrefs) == 0:
                continue
            if tp == "string":
                content = "pre_strs+" + content
                for func_n, addr in func_xrefs.items():
                    if content in str_dict:
                        if func_n not in str_dict[content]:
                            str_dict[content].append((func_n, addr))
                    else:
                        str_dict[content] = [(func_n, addr)]
            else:
                content = "pre_vars+" + content
                for func_n, addr in func_xrefs.items():
                    if content in var_dict:
                        if func_n not in var_dict[content]:
                            var_dict[content].append((func_n, addr))
                    else:
                        var_dict[content] = [(func_n, addr)]
    return var_dict, str_dict

def check_is_auto_var(var):
    var = var[9:].split("|||")[0]
    auto_list = ["word", "byte", "var", "jpt", "loc", "stru", "off", "flt", "dbl"]
    flag = 0
    if "." in var and len(var.split(".")) == 2:
        size, addr = var.split(".")
        try:
            addr = int(addr, 16)
            return True
        except:
             return False

    if "_" in var and len(var.split("_")) == 2:
        size, addr = var.split("_")
        for al in auto_list:
            if al in size:
                flag += 1
                break
        try:
            addr = int(addr, 16)
            flag += 1
        except:
             return False
        if flag >= 2:
            return True
    return False

def get_func_block_num(func_ea):
    if ida_ver >= 7.0:
        return len([(v.start_ea, v.end_ea) for v in FlowChart(Get_func(func_ea))])
    else:
        return len([(v.startEA, v.endEA) for v in FlowChart(Get_func(func_ea))])

# delete the first '.' in the function name
def get_unified_funcname(ea):
    funcname = GetFuncName(ea)
    if len(funcname) > 0:
        if '.' == funcname[0]:
            funcname = funcname[1:]
    return funcname



def combine_data(path, save_path):
    ans = {}
    for file in os.listdir(path):
        filep = os.path.join(path, file)
        fp = file.split(".json")[0]
        with open(filep, "r") as f:
            data = json.load(f)
        for fname, ds in data.items():
            fn = fp + "|||" + fname
            if fn not in ans:
                ans[fn] = ds
            else:
                ans[fn].extend(ds)
    with open(os.path.join(save_path, "all_" + path.split("/")[-1].lower() + ".json"), "w") as f:
        json.dump(ans, f)
    pass

def auto_analysis():
    # add the path of the data_base
    data_base = DATA_BASE
    filePath = os.path.join(data_base, "EESG/")
    if not os.path.exists(filePath):
        os.makedirs(filePath)
    inputName = GetInputFilePath()
    fname_sp = inputName.split('/')[-1].split(".elf")[0]
    fileName = filePath + fname_sp + '.pkl'

    # save the strings, global variables and external functions
    strings_path = os.path.join(data_base, "Strings/")
    if not os.path.exists(strings_path):
        os.makedirs(strings_path)
    global_var_path = os.path.join(data_base, "Global_vars/")
    if not os.path.exists(global_var_path):
        os.makedirs(global_var_path)
    ef_path = os.path.join(data_base, "External_functions/")
    if not os.path.exists(ef_path):
        os.makedirs(ef_path)
    s_name = strings_path + fname_sp + '.json'
    g_name = global_var_path + fname_sp + '.json'
    e_name = ef_path + fname_sp + '.json'
    s_dict = {}
    g_dict = {}
    e_dict = {}

    # begin to extract
    Wait()
    callees = dict()
    func_addr_dict = dict()
    data_segs = ['.rodata', '.bss', '.data', '.idata']
    functions = set()
    extern_func = set()
    min_block_num = 1

    for function_ea in AllFunctions():
        f_name = get_unified_funcname(function_ea)
        if len(f_name) == 0:
            continue
        if is_func_in_text(function_ea):
            functions.add(f_name)
            func_addr_dict[f_name] = function_ea
            for ref_ea in CodeRefsTo(function_ea, 0):
                if is_func_in_text(ref_ea):
                    caller_name = get_unified_funcname(ref_ea)
                    if len(caller_name) == 0:
                        continue
                    callees[caller_name] = callees.get(caller_name, set())
                    callees[caller_name].add(f_name)
        if is_func_in_extern(function_ea):
            extern_func.add(function_ea)
            func_addr_dict[f_name] = function_ea

    g = nx.MultiDiGraph()
    for f in functions:
        if len(f) != 0:
            blocks = get_func_block_num(func_addr_dict[f])
            if blocks >= min_block_num:
                g.add_node(f, func_addr=str(func_addr_dict[f]), type="user function", block_num=str(blocks))
            if f in callees:
                for f2 in callees[f]:
                    blocks = get_func_block_num(func_addr_dict[f2])
                    if blocks >= min_block_num:
                        g.add_node(f2, func_addr=str(func_addr_dict[f2]), type="user function", block_num=str(blocks))
                        g.add_edge(f, f2, rel_type="call")
                        g.add_edge(f2, f, rel_type="be_called")  

    callers = {}
    for extern_f in extern_func:
        f_name = get_unified_funcname(extern_f)
        if len(f_name) == 0:
            continue
        for ref_ea in CodeRefsTo(extern_f, 0):
            if is_func_in_plt(ref_ea):
                ref_ea = GetFuncAttr(ref_ea, idc.FUNCATTR_START)
                for ref_e in CodeRefsTo(ref_ea, 0):
                    if is_func_in_text(ref_e):
                        caller_name = get_unified_funcname(ref_e)
                        if caller_name not in functions or len(caller_name) == 0:
                            continue
                        callers[f_name] = callers.get(f_name, set())
                        callers[f_name].add(caller_name)
            else:
                if is_func_in_text(ref_ea):
                    caller_name = get_unified_funcname(ref_ea)
                    if caller_name not in functions or len(caller_name) == 0:
                        continue
                    callers[f_name] = callers.get(f_name, set())
                    callers[f_name].add(caller_name)
    for f in callers.keys():
        if len(f) != 0:
            g.add_node(f, func_addr=str(func_addr_dict[f]), type="external function")
            for f2 in callers[f]:
                if f2 not in e_dict:
                    e_dict[f2] = [f]
                else:
                    e_dict[f2].append(f)
                g.add_edge(f2, f, rel_type="external_call")
                g.add_edge(f, f2, rel_type="external_be_called")

    var_dict, str_dict = extract_global_var(data_segs, functions)
   
    for var in var_dict.keys():
        if len(var) != 0:
            flag = 0
            for f2 in var_dict[var]:
                f2_name = f2[0]
                blocks = get_func_block_num(f2[1])
                if f2_name not in functions or blocks < min_block_num:
                    continue
                if flag == 0:
                    if "|||auto_var" not in var and len(var) > 3:
                        g.add_node(var, type="const variable")
                    flag = 1
                if "|||auto_var" not in var and len(var) > 3:
                    if f2_name not in g_dict:
                        g_dict[f2_name] = [var]
                    else:
                        g_dict[f2_name].append(var)
                    g.add_edge(f2_name, var, rel_type="var_use")
                    g.add_edge(var, f2_name, rel_type="var_be_used")
                for f3 in var_dict[var]:
                    f3_name = f3[0]
                    blocks = get_func_block_num(f3[1])
                    if blocks < min_block_num or f3_name not in functions:
                        continue
                    if f2_name != f3_name:
                        g.add_edge(f2_name, f3_name, rel_type="variable_dependency")
                        g.add_edge(f3_name, f2_name, rel_type="variable_dependency_loop")

    for var in str_dict.keys():
        if len(var) != 0:
            flag = 0
            for f2 in str_dict[var]:
                f2_name = f2[0]
                blocks = get_func_block_num(f2[1])
                if f2_name not in functions or blocks < min_block_num:
                    continue
                if flag == 0:
                    if len(var) >= 5:
                        g.add_node(var, type="const strings")
                    flag = 1
                if len(var) >= 5:
                    g.add_edge(f2_name, var, rel_type="string_use")
                    g.add_edge(var, f2_name, rel_type="string_be_used")
                    if f2_name not in s_dict:
                        s_dict[f2_name] = [var]
                    else:
                        s_dict[f2_name].append(var)
                for f3 in str_dict[var]:
                    f3_name = f3[0]
                    blocks = get_func_block_num(f3[1])
                    if f3_name not in functions or blocks < min_block_num:
                        continue
                    if f2_name != f3_name:
                        g.add_edge(f2_name, f3_name, rel_type="string_dependency")
                        g.add_edge(f3_name, f2_name, rel_type="string_dependency_loop")

    addr_func = {}
    for func in functions:
        if len(func) != 0:
            addr = func_addr_dict[func]
            addr_func[addr] = func
    sorted_list = sorted([(addr, fn) for addr, fn in addr_func.items()], reverse=True)
    for si1 in range(0, len(sorted_list) - 1):
        sl1 = sorted_list[si1]
        sl2 = sorted_list[si1 + 1]
        blocks1 = get_func_block_num(sl1[0])
        blocks2 = get_func_block_num(sl2[0])
        if blocks2 < min_block_num or blocks1 < min_block_num:
            continue
        g.add_edge(sl1[1], sl2[1], rel_type="address_after")
        g.add_edge(sl2[1], sl1[1], rel_type="address_before")
    if len(functions) > 0:
        nx.write_gpickle(g, fileName)
    with open(s_name, "w") as f:
        json.dump(s_dict, f)
    with open(g_name, "w") as f:
        json.dump(g_dict, f)
    with open(e_name, "w") as f:
        json.dump(e_dict, f)
    combine_data(strings_path, data_base)
    combine_data(global_var_path, data_base)
    combine_data(ef_path, data_base)
    Exit(0)

auto_analysis()
