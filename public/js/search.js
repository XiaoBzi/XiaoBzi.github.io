

summaryInclude=60;
var fuseOptions = {
shouldSort: true,
includeMatches: true,
threshold: 0.0,
tokenize:true,
location: 0,
distance: 100,
maxPatternLength: 32,
minMatchCharLength: 1,
keys: [
{name:"title",weight:0.8},
{name:"contents",weight:0.5},
{name:"tags",weight:0.3},
{name:"categories",weight:0.3}
]
};


var searchQuery = param("s");
if(searchQuery){
$("#search-query").val(searchQuery);
executeSearch(searchQuery);
}else {
$('#search-results').append("<p></p>");
}

// 添加事件监听器到查找按钮
document.getElementById('search-button').addEventListener('click', function() {
    var searchQuery = document.getElementById('search-query').value;
    if (searchQuery) {
      executeSearch(searchQuery);
    }
  });

// 监听输入框的focus事件
document.getElementById('search-query').addEventListener('focus', function() {
    // 如果输入框的值是默认提示，则清空它
    if (this.value === this.placeholder) {
      this.value = '';
    }
    // 移除默认提示样式
    this.style.color = 'black';
  });
  
  // 监听输入框的blur事件
  document.getElementById('search-query').addEventListener('blur', function() {
    // 如果输入框为空，则显示默认提示
    if (!this.value.length) {
      this.value = this.placeholder;
      // 恢复默认提示样式
      this.style.color = 'gray';
    }
  });

  // 添加事件监听器到清空屏幕按钮
document.getElementById('reset-button').addEventListener('click', function() {
    // 清空搜索框
    document.getElementById('search-query').value = '';
    // 清空搜索结果
    document.getElementById('search-results').innerHTML = '<h3>搜索结果</h3>';
    // 重置搜索框样式（如果需要）
    document.getElementById('search-query').style.color = 'gray';
  });
  
  // 添加事件监听器到查找按钮
  document.getElementById('search-button').addEventListener('click', function() {
    var searchQuery = document.getElementById('search-query').value;
    if (searchQuery) {
      executeSearch(searchQuery);
    } else {
      // 如果搜索框为空，也清空搜索结果
      document.getElementById('search-results').innerHTML = '<h3>搜索结果</h3>';
    }
  });

  function executeSearch(searchQuery){
    $.getJSON( "/index.json", function( data ) {
      var pages = data;
      var fuse = new Fuse(pages, fuseOptions);
      var result = fuse.search(searchQuery);
      console.log({"matches":result});
      if(result.length > 0){
        populateResults(result);
      }else{
        $('#search-results').html("<h3>搜索结果</h3><p>小编未找到相关内容，请检查是否有输入错误，或者添加更详细的搜索关键词。</p>");
        $('#search-query').val(''); // 清空搜索框
      }
    });
  }

function populateResults(result){
$.each(result,function(key,value){
var contents= value.item.content;
var contents1= value.item.contents;
console.log({"matchescontent":contents});
console.log({"mcontentssssss":contents1});
var snippet = "";
var snippetHighlights=[];
var tags =[];
if( fuseOptions.tokenize ){
snippetHighlights.push(searchQuery);
}else{
$.each(value.matches,function(matchKey,mvalue){
if(mvalue.key == "tags" || mvalue.key == "categories" ){
snippetHighlights.push(mvalue.value);
}else if(mvalue.key == "content"){
start = mvalue.indices[0][0]-summaryInclude>0?mvalue.indices[0][0]-summaryInclude:0;
end = mvalue.indices[0][1]+summaryInclude<contents.length?mvalue.indices[0][1]+summaryInclude:contents.length;
snippet += contents.substring(start,end);
snippetHighlights.push(mvalue.value.substring(mvalue.indices[0][0],mvalue.indices[0][1]-mvalue.indices[0][0]+1));
}
});
}

    if(snippet.length<1){
       // 检查 contents 是否被定义
  if (contents !== undefined) {
    snippet += contents.substring(0, summaryInclude * 2);
  } else if (contents1 !== undefined) {
    // 如果 contents 未定义，尝试使用 contents1
    snippet += contents1.substring(0, summaryInclude * 2);
  } else {
    // 如果两者都未定义，跳过添加摘要的步骤
    snippet = ''; // 或者设置为其他默认值，或者进行其他处理
  }
    }
    //pull template from hugo templarte definition
    var templateDefinition = $('#search-result-template').html();
    //replace values
    var output = render(templateDefinition,{key:key,title:value.item.title,link:value.item.permalink,tags:value.item.tags,categories:value.item.categories,snippet:snippet});
    $('#search-results').append(output);

    $.each(snippetHighlights,function(snipkey,snipvalue){
      $("#summary-"+key).mark(snipvalue);
    });

});
}

function param(name) {
return decodeURIComponent((location.search.split(name + '=')[1] || '').split('&')[0]).replace(/\+/g, ' ');
}

function render(templateString, data) {
var conditionalMatches,conditionalPattern,copy;
conditionalPattern = /\$\{\s*isset ([a-zA-Z]*) \s*\}(.*)\$\{\s*end\s*}/g;
//since loop below depends on re.lastInxdex, we use a copy to capture any manipulations whilst inside the loop
copy = templateString;
while ((conditionalMatches = conditionalPattern.exec(templateString)) !== null) {
if(data[conditionalMatches[1]]){
//valid key, remove conditionals, leave contents.
copy = copy.replace(conditionalMatches[0],conditionalMatches[2]);
}else{
//not valid, remove entire section
copy = copy.replace(conditionalMatches[0],'');
}
}
templateString = copy;
//now any conditionals removed we can do simple substitution
var key, find, re;
for (key in data) {
find = '\\$\\{\\s*' + key + '\\s*\\}';
re = new RegExp(find, 'g');
templateString = templateString.replace(re, data[key]);
}
return templateString;
}

