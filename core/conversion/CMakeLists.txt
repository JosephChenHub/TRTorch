
file (GLOB_RECURSE CXX_SRCS *.cc *.cpp *.c)

set(HEADERS conversionctx converters evaluators tensorcontainer var)


set(SUB_SRCS ${CXX_SRCS} PARENT_SCOPE)
set(SUB_INC ${HEADERS} PARENT_SCOPE) 

