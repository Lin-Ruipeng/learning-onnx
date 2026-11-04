"""各章脚本共用的工具。

只在脚本里 import 具体模块（``from common.paths import outputs``），
这里不做转发式的 re-export，免得 ``__init__`` 变成第二个 utils。
"""
