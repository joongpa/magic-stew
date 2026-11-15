#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Buffer-only ABI: no NumPy headers or dependency on its binary ABI. */
static int buffer(PyObject *obj, Py_buffer *view, int ndim, Py_ssize_t itemsize,
                  char kind) {
    if (PyObject_GetBuffer(obj, view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) return 0;
    const char *format = view->format ? view->format : "";
    int valid = kind == 'i' ? (!strcmp(format,"l") || !strcmp(format,"q") || !strcmp(format,"n")) :
                kind == 'b' ? (!strcmp(format,"B") || !strcmp(format,"?")) : !strcmp(format,"d");
    if (view->ndim != ndim || view->itemsize != itemsize || !valid) {
        PyErr_SetString(PyExc_ValueError, "kernel requires contiguous native arrays with matching shape and dtype");
        return 0;
    }
    return 1;
}

static void release(Py_buffer *views, int count) {
    for (int i=0; i<count; ++i) if (views[i].obj) PyBuffer_Release(&views[i]);
}

static PyObject *permute(PyObject *self, PyObject *args) {
    PyObject *state, *sources;
    Py_buffer v={0};
    if (!PyArg_ParseTuple(args,"OO",&state,&sources)) return NULL;
    if (!PyBytes_Check(state) || PyBytes_GET_SIZE(state)!=54) {
        PyErr_SetString(PyExc_ValueError,"state must contain 54 bytes"); return NULL;
    }
    if (!buffer(sources,&v,1,sizeof(Py_ssize_t),'i')) { release(&v,1); return NULL; }
    if (v.shape[0]!=54) {
        PyErr_SetString(PyExc_ValueError,"permutation must contain 54 indices"); release(&v,1); return NULL;
    }
    const Py_ssize_t *p=v.buf;
    char data[54], *input=PyBytes_AS_STRING(state);
    for (int i=0;i<54;++i) {
        if (p[i]<0 || p[i]>=54) {
            PyErr_SetString(PyExc_ValueError,"permutation index out of range"); release(&v,1); return NULL;
        }
        data[i]=input[p[i]];
    }
    release(&v,1);
    return PyBytes_FromStringAndSize(data,54);
}

static PyObject *permute_many(PyObject *self, PyObject *args) {
    PyObject *state,*sources,*result=NULL;
    Py_buffer v={0};
    if (!PyArg_ParseTuple(args,"OO",&state,&sources)) return NULL;
    if (!PyBytes_Check(state) || PyBytes_GET_SIZE(state)!=54) {
        PyErr_SetString(PyExc_ValueError,"state must contain 54 bytes"); return NULL;
    }
    if (!buffer(sources,&v,2,sizeof(Py_ssize_t),'i')) goto fail;
    if (v.shape[1]!=54) { PyErr_SetString(PyExc_ValueError,"permutations must have width 54"); goto fail; }
    const Py_ssize_t *p=v.buf;
    for (Py_ssize_t i=0;i<v.shape[0]*54;++i) if (p[i]<0 || p[i]>=54) {
        PyErr_SetString(PyExc_ValueError,"permutation index out of range"); goto fail;
    }
    result=PyList_New(v.shape[0]);
    if (!result) goto fail;
    const char *input=PyBytes_AS_STRING(state);
    for (Py_ssize_t row=0;row<v.shape[0];++row) {
        char data[54];
        for (int j=0;j<54;++j) data[j]=input[p[row*54+j]];
        PyObject *value=PyBytes_FromStringAndSize(data,54);
        if (!value) goto fail;
        PyList_SET_ITEM(result,row,value);
    }
    release(&v,1); return result;
fail:
    release(&v,1); Py_XDECREF(result); return NULL;
}

/* NumPy sorts NaNs after finite values; equal keys retain the first row. */
static int compare(double a,double b) {
    if (isnan(a)) return isnan(b) ? 0 : 1;
    if (isnan(b)) return -1;
    return (a>b)-(a<b);
}

static PyObject *best_rows(PyObject *self, PyObject *args) {
    PyObject *objects[5],*result=NULL;
    Py_buffer v[5]={{0}};
    Py_ssize_t *best=NULL;
    if (!PyArg_ParseTuple(args,"OOOOO",&objects[0],&objects[1],&objects[2],&objects[3],&objects[4])) return NULL;
    for (int k=0;k<5;++k) if (!buffer(objects[k],&v[k],1,
            k==1 ? sizeof(Py_ssize_t) : sizeof(double), k==1 ? 'i' : 'd')) goto fail;
    Py_ssize_t n=v[0].shape[0];
    for (int k=1;k<5;++k) if (v[k].shape[0]!=n) {
        PyErr_SetString(PyExc_ValueError,"rank arrays must have equal length"); goto fail;
    }
    const double *cost=v[0].buf,*inliers=v[2].buf,*p90=v[3].buf,*layer=v[4].buf;
    const Py_ssize_t *groups=v[1].buf;
    Py_ssize_t count=0;
    for (Py_ssize_t i=0;i<n;++i) {
        if (groups[i]<0 || groups[i]>=n) {
            PyErr_SetString(PyExc_ValueError,"group IDs must be dense nonnegative row indices"); goto fail;
        }
        if (groups[i]>=count) count=groups[i]+1;
    }
    best=PyMem_Malloc((size_t)(count ? count : 1)*sizeof(Py_ssize_t));
    if (!best) { PyErr_NoMemory(); goto fail; }
    for (Py_ssize_t g=0;g<count;++g) best[g]=-1;
    for (Py_ssize_t i=0;i<n;++i) {
        if (!isfinite(cost[i])) continue;
        Py_ssize_t old=best[groups[i]];
        int cmp=old<0 ? -1 : compare(cost[i],cost[old]);
        if (old>=0 && !cmp) cmp=compare(-inliers[i],-inliers[old]);
        if (old>=0 && !cmp) cmp=compare(p90[i],p90[old]);
        if (old>=0 && !cmp) cmp=compare(-layer[i],-layer[old]);
        if (cmp<0) best[groups[i]]=i;
    }
    result=PyList_New(0);
    if (!result) goto fail;
    /* Output in original row order, as the existing scorer does. */
    for (Py_ssize_t i=0;i<n;++i) if (best[groups[i]]==i) {
        PyObject *index=PyLong_FromSsize_t(i);
        if (!index || PyList_Append(result,index)<0) { Py_XDECREF(index); goto fail; }
        Py_DECREF(index);
    }
    PyMem_Free(best); release(v,5); return result;
fail:
    PyMem_Free(best); release(v,5); Py_XDECREF(result); return NULL;
}

static int popcount(uint64_t value) {
    int count=0;
    while (value) { value &= value-1; ++count; }
    return count;
}

static PyObject *color_metrics(PyObject *self, PyObject *args) {
    PyObject *objects[4],*outputs[5]={NULL};
    Py_buffer v[4]={{0}};
    double confusion;
    if (!PyArg_ParseTuple(args,"OOOOd",&objects[0],&objects[1],&objects[2],&objects[3],&confusion)) return NULL;
    for (int k=0;k<4;++k) if (!buffer(objects[k],&v[k],2,k==3?sizeof(double):1,k==3?'d':'b')) goto fail;
    Py_ssize_t states=v[0].shape[0], configs=v[1].shape[0], obs=v[1].shape[1];
    if (v[0].shape[1]!=54 || v[2].shape[0]!=configs || v[2].shape[1]!=obs ||
            v[3].shape[0]!=54 || v[3].shape[1]!=obs || (configs && states>PY_SSIZE_T_MAX/configs/8)) {
        PyErr_SetString(PyExc_ValueError,"invalid batched color array shapes"); goto fail;
    }
    const uint8_t *identities=v[0].buf,*nearest=v[1].buf,*valid=v[2].buf;
    const double *costs=v[3].buf;
    for (Py_ssize_t i=0;i<states*54;++i) if (identities[i]>=54) {
        PyErr_SetString(PyExc_ValueError,"identity index out of range"); goto fail;
    }
    for (Py_ssize_t i=0;i<configs*obs;++i) if (nearest[i]>=54) {
        PyErr_SetString(PyExc_ValueError,"sticker index out of range"); goto fail;
    }
    for (int k=0;k<5;++k) {
        outputs[k]=PyBytes_FromStringAndSize(NULL,states*configs*(k?sizeof(int32_t):sizeof(double)));
        if (!outputs[k]) goto fail;
    }
    double *errors=(double *)PyBytes_AS_STRING(outputs[0]);
    int32_t *correct=(int32_t *)PyBytes_AS_STRING(outputs[1]),*confused=(int32_t *)PyBytes_AS_STRING(outputs[2]),
            *forbidden=(int32_t *)PyBytes_AS_STRING(outputs[3]),*totals=(int32_t *)PyBytes_AS_STRING(outputs[4]);
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t c=0;c<configs;++c) {
        uint64_t occupied=0;
        for (Py_ssize_t o=0;o<obs;++o) if (valid[c*obs+o]) occupied |= UINT64_C(1)<<nearest[c*obs+o];
        int total=popcount(occupied);
        for (Py_ssize_t s=0;s<states;++s) {
            uint64_t exact=0,permitted=0;
            for (Py_ssize_t o=0;o<obs;++o) if (valid[c*obs+o]) {
                uint8_t position=nearest[c*obs+o];
                double cost=costs[identities[s*54+position]*obs+o];
                if (cost==0.) exact |= UINT64_C(1)<<position;
                if (cost<1.) permitted |= UINT64_C(1)<<position;
            }
            Py_ssize_t i=s*configs+c;
            correct[i]=popcount(exact);
            confused[i]=popcount(permitted)-correct[i];
            forbidden[i]=total-correct[i]-confused[i]; totals[i]=total;
            errors[i]=total ? (confusion*confused[i]+forbidden[i])/total : (obs?INFINITY:0.);
        }
    }
    Py_END_ALLOW_THREADS
    release(v,4);
    return Py_BuildValue("NNNNN",outputs[0],outputs[1],outputs[2],outputs[3],outputs[4]);
fail:
    release(v,4); for (int k=0;k<5;++k) Py_XDECREF(outputs[k]); return NULL;
}

static PyMethodDef methods[]={
    {"permute",permute,METH_VARARGS,"Apply an exact 54-sticker permutation."},
    {"permute_many",permute_many,METH_VARARGS,"Build a batch of exact successor states."},
    {"best_rows",best_rows,METH_VARARGS,"Stable grouped minimum with the existing rank keys."},
    {"color_metrics",color_metrics,METH_VARARGS,"Batch exact, confused and forbidden sticker counts."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_search_kernels",NULL,-1,methods};
PyMODINIT_FUNC PyInit__search_kernels(void) { return PyModule_Create(&module); }
